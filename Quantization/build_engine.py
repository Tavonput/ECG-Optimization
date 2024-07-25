import os
import logging

from dataclasses import dataclass

import numpy as np

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from .image_batcher import ImageBatcher

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("TRTEG").setLevel(logging.INFO)
log = logging.getLogger("TRTEG")


#===========================================================================================================================
# Calibration
#
class EngineCalibrator(trt.IInt8Calibrator):
    """
    Engine Calibrator

    Implementation of TensorRT IInt8Calibrator.
    """
    def __init__(
        self, 
        algorithm:  trt.CalibrationAlgoType,
        cache_dir:  str,
    ) -> None:
        super().__init__()

        self.batch_allocation = None
        self.algorithm        = algorithm
        self.cache_dir        = cache_dir

        self.image_batcher = None

    
    def set_image_batcher(self, image_batcher: ImageBatcher) -> None:
        """
        Set the image batcher and allocate device memory for batching.

        Parameters
        ----------
        image_batcher : ImageBatcher
            The ImageBatcher.
        """
        log.info(f"Setting image batcher and allocating {image_batcher.nbytes} bytes of device memory")
        self.image_batcher    = image_batcher
        self.batch_allocation = cuda.mem_alloc(image_batcher.nbytes)


    def get_algorithm(self) -> trt.CalibrationAlgoType:
        """
        Overrides from TensorRT IInt8Calibrator. 
        Get the algorithm used for calibration.

        Returns
        -------
        algo : trt.CalibrationAlgoType
            Calibration algorithm.
        """
        return self.algorithm
    

    def get_batch(self, names: list[str]) -> list[int]:
        """
        Overrides from TensorRT IInt8Calibrator.

        Parameters
        ----------
        names : list[str]
            Names of inputs. Not Used.

        Returns
        -------
        allocations : list[int]
            List of device memory pointers.
        """
        if not self.image_batcher:
            return None
        
        try:
            batch = next(self.image_batcher)

            assert batch.nbytes == self.image_batcher.nbytes, f"{batch.nbytes} != {self.image_batcher.nbytes}"
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch.flat))

            return [int(self.batch_allocation)]
        
        except StopIteration:
            log.info("Calibration batches finished")
            return None
    

    def get_batch_size(self) -> int:
        """
        Overrides from TensorRT IInt8Calibrator. 
        Get the batch size.

        Returns
        -------
        size : int
            Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1
    

    def read_calibration_cache(self) -> bytes:
        """
        Overrides from TensorRT IInt8Calibrator. 
        Read the calibration cache file.

        Returns
        -------
        stream : bytes
            Byte stream of calibration cache contents.
        """
        if os.path.exists(self.cache_dir):
            with open(self.cache_dir, "rb") as file:
                log.info(f"Using calibration cache file {self.cache_dir}")
                return file.read()
    

    def write_calibration_cache(self, cache: bytes) -> None:
        """
        Overrides from TensorRT IInt8Calibrator.
        Write calibration cache to disk.

        Parameters
        ----------
        cache : bytes
            Byte stream content of the calibration cache.
        """
        cache_root_dir = os.path.dirname(self.cache_dir)
        if not os.path.exists(cache_root_dir):
            os.makedirs(cache_root_dir)

        with open(self.cache_dir, "wb") as file:
            log.info(f"Writing calibration cache to {self.cache_dir}")
            file.write(cache)


    def display_internal(self) -> None:
        log.info(f"Engine Calibrator... \n\
            \tImage Batcher: {self.image_batcher}\n\
            \tAlgorithm: {self.algorithm}\n\
            \tCache: {self.cache_dir}\n\
            \tBatch Size: {self.get_batch_size()}"
        )
        

#===========================================================================================================================
# Engine
#
@dataclass
class CalibrationConfig:
    dataset:     str
    cache:       str
    image_size:  int
    batch_size:  int
    max_batches: int
    algorithm:   str


def _parse_onnx(network: trt.INetworkDescription, logger: trt.ILogger, onnx_path: str) -> None:
    """
    Helper for parsing the ONNX model.

    Parameters
    ----------
    network : trt.INetworkDescription
        The TensorRT network that the ONNX model will be loaded into.
    logger : trt.ILogger
        The TensorRT logger.
    onnx_path : str
        The path to the ONNX model.
    """
    parser = trt.OnnxParser(network, logger)

    log.info(f"Parsing ONNX model {onnx_path}")
    with open(onnx_path, "rb") as onnx_model:
        if not parser.parse(onnx_model.read()):
            for i in range(parser.num_errors):
                log.error(parser.get_error(i))
    log.info(f"Parsing complete")

    inputs  = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    log.info("Network description...")
    for input in inputs:
        log.info(f"Name: {input.name} - Type: {input.dtype} - Shape: {input.shape}")
    for output in outputs:
        log.info(f"Name: {output.name} - Type: {output.dtype} - Shape: {output.shape}")


def _set_precision(
    builder:      trt.Builder, 
    config:       trt.IBuilderConfig, 
    network:      trt.INetworkDescription, 
    precision:    str, 
    calib_config: CalibrationConfig = None
) -> None:
    """
    Helper for setting the precision implicitly.

    Parameters
    ----------
    builder : trt.Builder
        The TensorRT builder.
    config : trt.IBuilderConfig
        The TensorRT config for setting the precision.
    network : trt.INetworkDescription
        The TensorRT Network. It should already be filled out by the ONNX parser.
    precision : str
        The precision to reduce to.
    calib_config : CalibrationConfig
        The CalibrationConfig for INT8.
    """
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    precision_flag = trt.DataType.FLOAT
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            log.warning(f"Platform does not support FP16. Falling back to FP32")
        else:
            log.info("Using FP16")
            precision_flag = trt.DataType.HALF
            config.set_flag(trt.BuilderFlag.FP16)
            
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            log.warning(f"Platform does not support INT8. Falling back to FP32")
        else:
            log.info("Using INT8")
            precision_flag = trt.DataType.INT8
            config.set_flag(trt.BuilderFlag.INT8)

            if calib_config.algorithm == "entropy":
                calib_algorithm = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
            elif calib_config.algorithm == "max":
                calib_algorithm = trt.CalibrationAlgoType.MINMAX_CALIBRATION

            config.int8_calibrator = EngineCalibrator(calib_algorithm, calib_config.cache)

            if not calib_config:
                log.error(f"INT8 wanted but no calibration config was given")

            if not os.path.exists(calib_config.cache):
                log.info("No calibration cache detected. Building image batcher for calibration")
                config.int8_calibrator.set_image_batcher(
                    ImageBatcher(
                        file_path  = calib_config.dataset,
                        batch_size = calib_config.batch_size,
                        image_size = calib_config.image_size,
                        shuffle    = True,
                        drop_last  = True,
                        max_batch  = calib_config.max_batches
                    )
                )
    
    # Manually set layers to the requested precision
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer.precision = precision_flag


def _inspect_engine(engine_path: str, logger: trt.Logger) -> None:
    """
    Log detailed engine description with an inspector. Saves everything to engine_inspector_log.txt.

    Parameters
    ----------
    engine_path : str
        Path to TensorRT engine.
    logger : trt.Logger
        TensorRT logger.
    """
    runtime = trt.Runtime(logger)
    assert runtime

    with open(engine_path, "rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    inspector = engine.create_engine_inspector()
    inspector.execution_context = context

    with open("engine_inspector_log.txt", "w") as file:
        file.write(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

    log.info("Engine inspector logged to engine_inspector_log.txt")


def build_engine(
    onnx_path:    str, 
    engine_path:  str, 
    precision:    str, 
    calib_config: CalibrationConfig = None
) -> None:
    """
    Build a TensorRT engine.

    Parameters
    ----------
    onnx_path : str
        Path to ONNX model.
    engine_path : str
        Path to save the engine.
    precision : str
        Model precision.
    calib_config : CalibrationConfig
        CalibrationConfig.

    TODO: 
    - Get dynamic batching to work. 
    - There is a CUDA memory access bug with high batch sizes during inference.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    config  = builder.create_builder_config()
    config.profiling_verbosity= trt.ProfilingVerbosity.DETAILED

    if precision == "explicit":
        log.info("Detected explicit precision")
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        network = builder.create_network()

    _parse_onnx(network, TRT_LOGGER, onnx_path)

    if precision != "explicit":
        _set_precision(builder, config, network, precision, calib_config)
            
    # Optimization profile
    # profile   = builder.create_optimization_profile()
    # min_shape = [1]  + input_shape
    # opt_shape = [32] + input_shape
    # max_shape = [64] + input_shape

    # profile.set_shape(inputs[0].name, min_shape, opt_shape, max_shape)
    # config.add_optimization_profile(profile)

    if builder.is_network_supported(network, config):
        log.info("Network is supported")
    else:
        log.error("Network is not supported")
        return

    # Build engine
    log.info(f"Building engine to {engine_path}")

    engine_dir = os.path.dirname(engine_path)
    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)

    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        log.error("Failed to build engine")
        return

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    log.info(f"Built engine to {engine_path}")

    _inspect_engine(engine_path, TRT_LOGGER)
    

#===========================================================================================================================
# Main (Used for testing this file)
#
if __name__ == "__main__":
    """
    Main function can be used for testing.
    """
    calibrator = EngineCalibrator(trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2, "Calibration-Cache/timing.cache")
    calibrator.display_internal()
