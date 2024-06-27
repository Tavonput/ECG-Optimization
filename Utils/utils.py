import time
import json
import os
from copy import deepcopy
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import *

from thop import profile

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ************************************************************************************************************************
# Training And Eval
#
def train(
    model:      nn.Module,
    dataloader: DataLoader,
    criterion:  nn.Module,
    optimizer:  Optimizer,
    scheduler:  LambdaLR
) -> None:
    """
    Train a model for one epoch.

    @param model: Model to train.
    @param dataloader: Training dataloader.
    @param criterion: Criterion.
    @param optimizer: Optimizer.
    @param scheduler: Scheduler.
    """
    model.train()

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        scheduler.step()


@torch.inference_mode()
def evaluate(
    model:      nn.Module,
    dataloader: DataLoader,
    verbose:    bool = True
) -> float:
    """
    Evaluate a model.

    @param model: Model to evaluate.
    @param dataloader: Testing dataloader.
    @param verbose: Verbosity.

    @return Accuracy.
    """
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, labels in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)

        num_samples += labels.size(0)
        num_correct += (outputs == labels).sum()
    
    return (num_correct / num_samples * 100).item()


def warm_up_dataloader(dataloader, num_batches: int = 0) -> None:
    """
    Warmup a dataloader by iterating over the batches and doing nothing. Can help with caching.

    @param dataloader: Dataloader to warmup.
    @param num_batches: Number of batches to load. If not specified, all batches will be used.
    """
    if num_batches == 0:
        for images, labels in tqdm(dataloader, desc="Warm-up", leave=False):
            continue
    else:
        i = 1
        for images, labels in enumerate(dataloader):
            if i >= num_batches:
                break
            i += 1


# ************************************************************************************************************************
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

    @param path: Path to json file.
    @param data: ModelStats instance to add.
    """
    print(f"[I]: Adding model stat to {path}")
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


def get_model_stats_from_json(path: str, model_names: list[str], sort: bool = False) -> list[ModelStats]:
    """
    Get a list of ModelStats instances from a json file by name searching.

    @param path: Path to json file.
    @param model_names: List of names corresponding to the ModelStats to retrieve.
    @param sort: Whether or not to sort the results based on the model_names.

    @return List of ModelStats.
    """
    if not os.path.exists(path):
        print(f"Path does not exist {path}")
        return None

    with open(path, "r") as file:
        try:
            database = json.load(file)
        except:
            print(f"There is no data in {path}")
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


def get_model_macs(model: nn.Module, inputs: torch.Tensor) -> int:
    """
    Get model MACs through thop profile.

    @param model: Model to profile.
    @param inputs: Dummy input.

    @return Number of MACs.
    """
    macs, _ = profile(model, inputs=(inputs,), verbose=False)
    return macs


def get_num_parameters(model: nn.Module) -> int:
    """
    Get the  number of parameters in a model.

    @param model: Model to profile:

    @return Number of parameters.
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

    @param model: Model to profile.
    @param dummy_input: Dummy input that will run through the model.
    @param n_warmup: Number of warmup iterations.
    @param n_test: Number of forward passes.
    @param test_device: Device to test on.

    @return Latency in seconds.

    TODO: Maybe GPU testing should be done separately to match the TensorRT latency benchmark.
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

    @param dataloader: Dataloader to profile.
    @param num_batches: Number of batches to test.

    @return Time to load one batch in seconds.
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

    @param model: Model to benchmark.
    @param dataloader: Testing dataloader.
    @param name: Name to be attached with the results.
    @param no_latency: Whether or not to skip latency profile.

    @return ModelStats instance.
    """
    print(f"[I]: Benchmarking model {name}")
    model.eval()
    model.to("cpu")

    dummy_input = next(iter(dataloader))[0][0].unsqueeze(0)

    print(f"[I]: \tGetting model params and MACs")
    macs    = get_model_macs(deepcopy(model).to(torch.float32), dummy_input.clone().to(torch.float32))
    params  = get_num_parameters(model)

    if no_latency:
        print(f"[I]: \tLatency measurement skipped")
        latency = 0
    else:
        print(f"[I]: \tMeasuring latency")
        latency = measure_latency(model, dummy_input)

    print(f"[I]: \tEvaluating")
    model.to(device)
    accuracy = evaluate(model, dataloader)

    print(f"[I]: Benchmarking for {name} finished")
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

    @param image_size: Image resolution.
    @param channels: Number of channels.
    @param num_images: Number of images.
    @param data_width: Bit size of each value.

    @return Dataset size in GiB.
    """
    pixels = image_size * image_size * channels
    bits   = pixels * data_width
    return (bits / GiB) * num_images


# ************************************************************************************************************************
# Display and Plotting
#
# TODO: Currently in the process of refactoring. Some function need to use the PlotConfig struct. Use subplot ax instead
#       of plt.
#
@dataclass
class PlotConfig:
    title:   str = "Title"
    x_label: str = "x-axis"
    y_label: str = "y-axis"

    x_range: tuple = None
    y_range: tuple = None

    x_scale: str = "linear"
    y_scale: str = "linear"

    x_grid: bool = True
    y_grid: bool = True

    fig_size: tuple = (10, 4)


def compare_models(
    models_stats: list[ModelStats], 
    show_macs:    bool = True, 
    show_params:  bool = True,
    fig_size:     tuple = (10, 4)
) -> None:
    """
    Compare a list of ModelStats displayed with a bar graphs.

    @param model_stats: List of ModelStats to compare.
    @param show_macs: Whether or not to show the MACs plot.
    @param show_params: Whether or not to show the params plot.
    @param fig_size: Figure size of the plots.
    """
    sns.set_style("whitegrid")

    names   = [model.name for model in models_stats]
    accs    = [model.accuracy for model in models_stats]
    macs    = [round(model.macs / 1e6) for model in models_stats]
    latency = [round(model.latency * 1000, 1) for model in models_stats]
    params  = [round(model.params / 1e6) for model in models_stats]

    plots = 2
    if show_macs: 
        plots += 1
    if show_params: 
        plots += 1

    fig, axs = plt.subplots(1, plots, figsize=fig_size)
    colors = sns.color_palette("husl", len(names))

    axs[0].bar(names, accs, color=colors)
    axs[0].set_title("Accuracy")
    acc_min_bound = np.clip(min(accs) - 10, 0, 100)
    acc_max_bound = np.clip(max(accs) + 5, 0, 100)
    axs[0].set_ylim([acc_min_bound, acc_max_bound])

    axs[1].bar(names, latency, color=colors)
    axs[1].set_title("Latency (ms)")

    if show_params:
        axs[2].bar(names, params, color=colors)
        axs[2].set_title("Parameters (M)")

    if show_macs:
        axs[3].bar(names, macs, color=colors)
        axs[3].set_title("MACs (M)")

    plt.tight_layout()
    plt.show()


def display_model_stats(model_stats: ModelStats) -> None:
    """
    Formatted display of a ModelStats instance.

    @param model_stats: ModelStats instance.
    """
    print(f"Name:     {model_stats.name}")
    print(f"Accuracy: {model_stats.accuracy:.2f}%")
    print(f"Latency:  {round(model_stats.latency * 1000, 1)} ms")
    print(f"Params:   {round(model_stats.params / 1e6)} M")
    print(f"MACs:     {round(model_stats.macs / 1e6)} M")


def compare_single_values(
    values:     dict[str, float | int],
    config:     PlotConfig,
    horizontal: bool = True,
) -> None:
    """
    Compare a list of single values through a bar graph.

    @param values: Dictionary of {names : values}.
    @param config: PlotConfig.
    @param horizontal: Whether or not to have a horizontal plot.
    """
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(values))

    labels = list(values.keys())
    values = list(values.values())

    _, ax = plt.subplots(figsize=config.fig_size)

    if horizontal:
        bars = ax.barh(labels[::-1], values[::-1], color=colors)
        ax.set_xlabel(config.x_label)
        ax.set_xscale(config.x_scale)
        ax.set_xlim(config.x_range)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}", 
                xy = (bar.get_y() + bar.get_width() / 2, height),
                xytext = (3, 0),
                textcoords = "offset points",
                ha = "center", va = "bottom", 
            )
    else:
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(config.y_label)
        ax.set_yscale(config.y_scale)
        ax.set_ylim(config.y_range)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}", 
                xy = (bar.get_x() + bar.get_width() / 2, height),
                xytext = (0, 3),
                textcoords = "offset points",
                ha = "center", va = "bottom", 
            )

    ax.set_title(config.title)
    ax.grid(config.x_grid, axis="x")
    ax.grid(config.y_grid, axis="y")

    plt.show()


def compare_list_values(
    values:  dict[str, list],
    x_axis:  str   = None,
    y_axis:  str   = None,
    title:   str   = None,
    y_range: tuple = None
) -> None:
    """
    Compare lists of values through a plot.

    @param values: Dictionary containing {"label": list of values}.
    @param x_axis: Label for x-axis.
    @param y_axis: Label for y_axis.
    @param title: Title.
    @param r_range: Range of display for the y-axis (min, max).
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    for k, v in values.items():
        plt.plot(v, label=k)

    if y_range:
        plt.ylim(y_range[0], y_range[1])
    if x_axis:
        plt.xlabel(x_axis)
    if y_axis:
        plt.ylabel(y_axis)
    if title:
        plt.title(title)
        
    plt.legend()
    plt.show()


def compare_pairwise(
    values: dict[str, list[tuple]],
    config: PlotConfig
) -> None:
    """
    Normal pairwise (x, y) plot.

    @param values: Dictionary containing labels and points {name : list of (x, y)}.
    @param config: PlotConfig.
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    _, ax = plt.subplots(figsize=config.fig_size)

    for k, v in values.items():
        xs, ys = zip(*v)
        ax.plot(xs, ys, "o-", label=k)

    if config.y_range:
        ax.set_ylim(config.y_range[0], config.y_range[1])    
    if config.x_range:
        ax.set_ylim(config.x_range[0], config.x_range[1])   

    ax.set_xlabel(config.x_label)
    ax.set_ylabel(config.y_label)
    ax.set_title(config.title)
    ax.legend()
    plt.show()
