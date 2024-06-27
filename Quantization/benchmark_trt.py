import time

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm

from .inference import *


Byte = 8
KiB  = 1024 * Byte
MiB  = 1024 * KiB
GiB  = 1024 * MiB


@dataclass
class ModelStats:
    name:     str   = "My Model"
    macs:     int   = 0
    params:   int   = 0
    latency:  float = 0
    accuracy: float = 0


def evaluate_trt(engine_path: str, dataloader: DataLoader) -> float:
    """
    Evaluate a TensorRt engine with a PyTorch dataloader.

    @param engine_path: Path to TensorRT engine.
    @param dataloader: Testing dataloader.

    @return Accuracy.
    """
    trt_context = create_trt_context(engine_path)
    
    # Setup inputs and outputs
    input_buffers, output_buffers = allocate_buffers(trt_context.engine, np.float32)
    for i in input_buffers:
        trt_context.context.set_tensor_address(i.name, int(i.device))
    for o in output_buffers:
        trt_context.context.set_tensor_address(o.name, int(o.device))
    trt_context.context.set_input_shape(input_buffers[0].name, input_buffers[0].shape)

    num_samples = 0
    num_correct = 0

    for inputs, labels in tqdm(dataloader, desc="Eval", leave=False):
        inputs = inputs.numpy()
        labels = labels.numpy()

        copy_data_to_host_buffer(input_buffers[0], inputs)
        inference(trt_context, input_buffers, output_buffers)
        
        output_buffers[0].host = output_buffers[0].host.reshape(output_buffers[0].shape)
        outputs = np.argmax(output_buffers[0].host, axis=1)

        num_samples += labels.shape[0]
        num_correct += np.sum(outputs == labels)
    
    return (num_correct / num_samples * 100).item()


def measure_latency(engine_path: str, n_warmup: int = 50, n_test: int = 50) -> float:
    """
    Measure the latency of a TensorRT engine. Latency testing is ran 10 times and then averaged.

    @param engine_path: Path to TensorRT engine.
    @param n_warmup: Number of warmup iterations.
    @param n_test: Number of forward passes.

    @return Latency in seconds.
    """
    trt_context = create_trt_context(engine_path)

    # Setup inputs and outputs
    input_buffers, output_buffers = allocate_buffers(trt_context.engine, np.float32)
    for i in input_buffers:
        trt_context.context.set_tensor_address(i.name, int(i.device))
    for o in output_buffers:
        trt_context.context.set_tensor_address(o.name, int(o.device))
    trt_context.context.set_input_shape(input_buffers[0].name, input_buffers[0].shape)

    dummy_input = np.random.rand(*input_buffers[0].shape)
    copy_data_to_host_buffer(input_buffers[0], dummy_input)
    [cuda.memcpy_htod_async(inp.device, inp.host, trt_context.stream) for inp in input_buffers]

    # Warmup
    for _ in range(n_warmup):
        inference(trt_context, input_buffers, output_buffers)

    # Real test
    times = []
    for _ in range(0, 10):
        t1 = time.time()

        for _ in range(0, n_test):
            trt_context.context.execute_async_v3(stream_handle=trt_context.stream.handle)
            trt_context.stream.synchronize()

        t2 = time.time()

        times.append((t2 - t1) / n_test)

    return sum(times) / len(times)


def benchmark_model_trt(engine_path: str, dataloader: DataLoader, name: str) -> ModelStats:
    """
    Benchmark a TensorRT engine for accuracy and latency. Does not set the parameter count or MACs.

    @param engine_path: Path to TensorRT engine.
    @param dataloader: Testing dataloader.
    @param name: Name to attach to results.

    @return ModelStats instance without param or macs set.
    """
    print(f"[I]: Benchmarking TensorRT model {name} from {engine_path}")

    print(f"[I]: \tEvaluating")
    accuracy = evaluate_trt(engine_path, dataloader)

    print(f"[I]: \tMeasuring latency")
    latencies = []
    for _ in range(0, 10):
        latencies.append(measure_latency(engine_path, n_test=100))
    latency = sum(latencies) / len(latencies)

    print(f"[I]: Benchmark for {name} finished")
    return ModelStats(
        name     = name,
        accuracy = accuracy,
        latency  = latency,
    )
