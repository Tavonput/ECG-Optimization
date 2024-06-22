import os

import tensorrt as trt


def inspect_engine(engine_path: str, logger: trt.Logger, input_shape: tuple) -> None:
    """
    Log detailed engine description with an inspector. Saves everything to engine_inspector_log.txt.

    @param engine_path: Path to TensorRT engine.
    @param logger: TensorRT logger.
    @param input_shape: Input shape.
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

    print("[INFO]: Engine inspector logged to engine_inspector_log.txt")


def build_engine(onnx_path, engine_path) -> None:
    """
    Build a TensorRT engine.

    @param onnx_path: Path to ONNX model.
    @param engine_path: Path to save the engine.

    TODO: 
    - Get dynamic batching to work. 
    - There is a CUDA memory access bug with high batch sizes during inference.
    - Allow choice of precision.
    - Do you need to use obey precision constrains?
    - INT8 quantization with calibration.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser  = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"[INFO]: Parsing ONNX model {onnx_path}")
    with open(onnx_path, "rb") as onnx_model:
        if not parser.parse(onnx_model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
    print(f"[INFO]: Parsing complete")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("[INFO]: Network description...")
    for input in inputs:
        print(f"\tName: {input.name}\n\t\tType: {input.dtype}\n\t\tShape: {input.shape}")
    for output in outputs:
        print(f"\tName: {output.name}\n\t\tType: {output.dtype}\n\t\tShape: {output.shape}")
    
    input_shape = inputs[0].shape[1:]

    # Config
    config = builder.create_builder_config()
    config.profiling_verbosity= trt.ProfilingVerbosity.DETAILED

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    
    # Optimization profile
    # profile   = builder.create_optimization_profile()
    # min_shape = [1]  + input_shape
    # opt_shape = [32] + input_shape
    # max_shape = [64] + input_shape

    # profile.set_shape(inputs[0].name, min_shape, opt_shape, max_shape)
    # config.add_optimization_profile(profile)

    # IO
    # inputs[0].allowed_formats = 1 << int(trt.TensorFormat.CHW16)
    # inputs[0].dtype  = trt.DataType.HALF
    # inputs[0].dtype = trt.DataType.HALF

    if builder.is_network_supported(network, config):
        print("[INFO]: Network is supported")
    else:
        print("[ERROR]: Network is not supported")
        return

    # Build engine
    print(f"[INFO]: Building engine to {engine_path}")

    engine_dir = os.path.dirname(engine_path)
    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[INFO]: Built engine to {engine_path}")

    # Inspect engine
    inspect_engine(engine_path, TRT_LOGGER, input_shape)
    