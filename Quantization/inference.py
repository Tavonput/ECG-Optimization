from dataclasses import dataclass

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np


@dataclass
class Buffer:
    host:   np.array
    device: int
    shape:  tuple
    size:   int
    nbytes: int
    name:   str


@dataclass
class TrtContext:
    runtime: trt.Runtime
    engine:  trt.ICudaEngine
    context: trt.IExecutionContext
    logger:  trt.Logger
    stream:  cuda.Stream


class DebugListener(trt.IDebugListener):
    """
    @class Debug Listener

    TensorRT IDebugListener implementation.
    """
    def __init__(self):
        super().__init__()

    def process_debug_tensor(self, addr, location, type, shape, name, stream):
        print(f"Name: {name}\n\tAddress: {addr}\n\t")


def create_trt_context(engine_path: str, log_level: trt.ILogger.Severity = trt.Logger.WARNING) -> TrtContext:
    """
    Setup all of the TensorRT stuff for inference.

    @param engine_path: Path to TensorRT engine.
    @param log_level: Severity level for logging. 

    @return TrtContext containing all inference objects.

    TODO: Don't use trt severity as an input. The user should not need to know about trt severity enums.
    """
    logger  = trt.Logger(log_level)

    runtime = trt.Runtime(logger)
    assert runtime

    with open(engine_path, "rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())
    assert engine

    context = engine.create_execution_context()
    assert context

    stream = cuda.Stream()

    context.set_optimization_profile_async(0, stream.handle)
    return TrtContext(
        runtime = runtime,
        engine  = engine,
        context = context,
        logger  = logger,
        stream  = stream,
    )


def allocate_buffers(engine: trt.ICudaEngine, data_type:np.dtype) -> tuple[list[Buffer], list[Buffer]]:
    """
    Allocate IO buffers. Host memory is pagelocked.

    @param engine: TensorRT engine.
    @param data_type: Numpy dtype used for memory allocation.

    @return IO buffers.
    """
    inputs = []
    outputs = []
    for binding in engine:
        # The max shape will be used
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            shape = engine.get_tensor_profile_shape(binding, 0)[-1]
        else:
            shape = engine.get_tensor_shape(binding)
            shape[0] = inputs[0].shape[0]
        size   = trt.volume(shape)
        nbytes = size * data_type().itemsize

        # Allocate memory
        host_mem   = cuda.pagelocked_empty(size, dtype=data_type)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Store buffers
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(Buffer(host_mem, device_mem, shape, size, nbytes, binding))
        else:
            outputs.append(Buffer(host_mem, device_mem, shape, size, nbytes, binding))
    
    return inputs, outputs


def copy_data_to_host_buffer(buffer: Buffer, data: np.array) -> None:
    """
    Copy data into the host memory of a buffer.

    @param buffer: Buffer to transfer data into.
    @param data: Data to transfer.
    """
    assert data.size == buffer.host.size
    np.copyto(buffer.host, np.ascontiguousarray(data.flat))


def inference(trt_cxt: TrtContext, inputs: list[Buffer], outputs: list[Buffer]) -> None:
    """
    Inference a TensorRT engine.

    @param trt_cxt: TrtContext built on the engine to inference.
    @param inputs: Input Buffers.
    @param outputs: Output Buffers
    """
    # Inputs: Host => Device
    [cuda.memcpy_htod_async(inp.device, inp.host, trt_cxt.stream) for inp in inputs]

    # Inference
    trt_cxt.context.execute_async_v3(stream_handle=trt_cxt.stream.handle)

    # Outputs: Device => Host
    [cuda.memcpy_dtoh_async(out.host, out.device, trt_cxt.stream) for out in outputs]

    trt_cxt.stream.synchronize()
