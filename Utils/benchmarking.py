import time
import json
import logging
import os
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thop import profile

import numpy as np

from .classification import evaluate

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("BENCH").setLevel(logging.INFO)
log = logging.getLogger("BENCH")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#===========================================================================================================================
# Benchmarking
#
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


def add_model_stat_to_json(path: str, data: ModelStats) -> None:
    """
    Append a ModelStats instance to a json that stores a list of ModelStats.

    Parameters
    ----------
    path : str
        Path to json file.
    data : ModelStats
        ModelStats instance to add.
    """
    log.info(f"Adding model stat to {path}")
    if not os.path.exists(path):
        with open(path, "w") as _:
            pass

    with open(path, "r") as file:
        try:
            database = json.load(file)
        except:
            database = []

    database.append(asdict(data))
    with open(path, "w") as file:
        json.dump(database, file, indent=4)


def get_model_stats_from_json(path: str, model_names: List[str], sort: bool = False) -> List[ModelStats]:
    """
    Get a list of ModelStats instances from a json file by name searching.

    Parameters
    ----------
    path : str
        Path to json file.
    model_names : List[str]
        List of names corresponding to the ModelStats to retrieve.
    sort : bool
        Whether or not to sort the results based on the model_names.

    Returns
    -------
    stats : List[ModelStats]
        List of ModelStats.
    """
    if not os.path.exists(path):
        log.error(f"Path does not exist {path}")
        return None

    with open(path, "r") as file:
        try:
            database = json.load(file)
        except:
            log.error(f"There is no data in {path}")
            return None
        
    # Get all model stats for the provided names
    model_stats = filter(lambda x: x["name"] in model_names, database)
    model_stats = map(lambda x: ModelStats(**x), model_stats)
    model_stats = list(model_stats)

    if sort:
        sorted_stats = []
        for model in model_names:
            for stat in model_stats:
                if stat.name == model:
                    sorted_stats.append(stat)
                    break
        model_stats = sorted_stats
    
    return model_stats

def get_model_size(param_count: int, bit_width: int) -> float:
    """
    Get the model size in MB.

    Parameters
    ----------
    param_count : int
        The number of parameters.
    bit_width : int
        The bit width.
    
    Returns
    -------
    size : float
        The size in MB.
    """
    return param_count * bit_width / MiB


def get_model_macs(model: nn.Module, inputs: torch.Tensor) -> int:
    """
    Get model MACs through thop profile.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    inputs : torch.Tensor
        Dummy input.

    Returns
    -------
    macs : int
        Number of MACs.
    """
    macs, _ = profile(model, inputs=(inputs,), verbose=False)
    return macs


def get_num_parameters(model: nn.Module) -> int:
    """
    Get the  number of parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Model to profile:

    Returns
    ------- 
    count : int
        Number of parameters.
    """
    num_counted_elements = 0
    for param in model.parameters():
        num_counted_elements += param.numel()
    return num_counted_elements


@torch.no_grad()
def measure_latency(
    model:       nn.Module, 
    dummy_input: torch.Tensor, 
    n_warmup:    int = 50, 
    n_test:      int = 100,
    test_device: str = "cpu",
) -> float:
    """
    Measure the latency of a model. Latency will be measured 10 times and averaged.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    dummy_input : torch.Tensor
        Dummy input that will run through the model.
    n_warmup : int
        Number of warmup iterations.
    n_test : int
        Number of forward passes.
    test_device : str
        Device to test on.

    Returns
    -------
    latency : float
        Latency in seconds.
    """
    model.eval()
    model.to(test_device)
    inp = dummy_input.to(test_device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(inp)

    # Real test
    times = []
    for _ in range(0, 10):
        t1 = time.time()
        for _ in range(n_test):
            _ = model(inp)
        t2 = time.time()
        times.append((t2 - t1) / n_test)
    
    return sum(times) / len(times)


def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 100) -> float:
    """
    Test how long it takes to load a batch from a dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to profile.
    num_batches : int
        Number of batches to test.

    Returns
    -------
    time : float
        Time to load one batch in seconds.
    """
    start_time = time.time()
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches - 1:
            break

    end_time = time.time()
    return (end_time - start_time) / num_batches


def benchmark_model(model: nn.Module, dataloader: DataLoader, name: str, no_latency: bool = False) -> ModelStats:
    """
    Benchmark a model for accuracy, latency, parameter count, and MACs.

    Parameters
    ----------
    model : nn.Module 
        Model to benchmark.
    dataloader : DataLoader
        Testing dataloader.
    name : str
        Name to be attached with the results.
    no_latency : bool
        Whether or not to skip latency profile.

    Return
    ------
    stats : ModelStats
        ModelStats instance.
    """
    log.info(f"Benchmarking model {name}")
    model.eval()
    model.to("cpu")

    dummy_input = next(iter(dataloader))[0][0].unsqueeze(0)

    log.info(f"\tGetting model params and MACs")
    macs    = get_model_macs(deepcopy(model).to(torch.float32), dummy_input.clone().to(torch.float32))
    params  = get_num_parameters(model)

    if no_latency:
        log.info(f"\tLatency measurement skipped")
        latency = 0
    else:
        log.info(f"\tMeasuring latency")
        latency = measure_latency(model, dummy_input)

    log.info(f"\tEvaluating")
    model.to(device)
    accuracy = evaluate(model, dataloader)

    log.info(f"Benchmarking for {name} finished")
    return ModelStats(
        macs     = macs, 
        params   = params, 
        latency  = latency, 
        accuracy = accuracy, 
        name     = name
    )


def get_dataset_size(image_size: int, channels: int, num_images: int, data_width: int) -> float:
    """
    Get the size of a dataset in GiB. Images must be square.

    Parameters
    ----------
    image_size : int
        Image resolution.
    channels : int
        Number of channels.
    num_images : int
        Number of images.
    data_width : int
        Bit size of each value.

    Returns
    -------
    size : float
        Dataset size in GiB.
    """
    pixels = image_size * image_size * channels
    bits   = pixels * data_width
    return (bits / GiB) * num_images


def compute_top_mean_std(values: np.ndarray, discard: int) -> Tuple[float, float, float]:
    """
    Compute the top value, mean, and std of a given array.

    Parameters
    ----------
    values : np.ndarray
        The array of values.
    discard : int
        How many initial values to discard for mean and std.

    Returns
    -------
    stats : float, float, float
        The top, mean, and std.
    """
    top  = max(values)
    mean = np.mean(values[discard:])
    std  = np.std(values[discard:])
    return top, mean, std


def time_to_convergence(accuracies: List[float], time_per_epoch: float, std_threshold: float, window_size: int) -> Tuple[int, float]:
    """
    Compute the time it takes for an array of accuracies to converge. Convergence is defined as...
        std(last_window_size_accuracies) <= std_threshold
        
    For example, if the window size is 5, then convergence is when the std of the last 5 runs is less than or equal to the
    std threshold.

    This function will return 0s if convergence was not met.

    Parameters
    ----------
    accuracies : List[float]
        The running accuracies.
    time_per_epoch : float
        The amount of time per epoch.
    std_threshold : float
        The std to compare against the window std.
    window_size : int
        How many previous entries should be considered during convergence testing.
    
    Returns
    -------
    epoch, time : int, float
        The epoch of convergence and the time to converge.
    """
    current_idx  = 0
    epoch        = 0
    while current_idx + window_size <= len(accuracies):
        window_std = np.std(accuracies[current_idx : (current_idx + window_size)])

        if window_std <= std_threshold:
            epoch = current_idx + window_size
            break

        current_idx += 1
    
    if epoch == 0:
        log.info(f"Accuracies did not converge")
        return 0, 0
    
    return epoch, epoch * time_per_epoch