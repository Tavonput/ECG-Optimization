import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import *
from torchprofile import profile_macs

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#######################################################################################################
# Training And Eval
#######################################################################################################
def train(
    model:      nn.Module,
    dataloader: DataLoader,
    criterion:  nn.Module,
    optimizer:  Optimizer,
    scheduler:  LambdaLR
) -> None:
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

def warm_up_dataloader(dataloader, num_batches: int = 0):
    if num_batches == 0:
        for images, labels in tqdm(dataloader, desc="Warm-up", leave=False):
            continue
    else:
        i = 1
        for images, labels in enumerate(dataloader):
            if i >= num_batches:
                break
            i += 1

#######################################################################################################
# Benchmarking
#######################################################################################################
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

@dataclass
class ModelStats:
    macs:     int
    params:   int
    latency:  float
    accuracy: float
    name:     str = "My Model"
    
def get_model_macs(model: nn.Module, inputs: torch.Tensor) -> int:
    return profile_macs(model, inputs)

def get_num_parameters(model: nn.Module) -> int:
    num_counted_elements = 0
    for param in model.parameters():
        num_counted_elements += param.numel()
    return num_counted_elements

@torch.no_grad()
def measure_latency(
    model:       nn.Module, 
    dummy_input: torch.Tensor, 
    n_warmup:    int = 50, 
    n_test:      int = 50,
) -> float:
    model.eval()

    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)

    # Real test
    times = []
    for _ in range(0, 10):
        t1 = time.time()
        for _ in range(n_test):
            _ = model(dummy_input)
        t2 = time.time()
        times.append((t2 - t1) / n_test)
    
    return sum(times) / len(times)

def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 100) -> float:
    start_time = time.time()
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches - 1:
            break

    end_time = time.time()
    return (end_time - start_time) / num_batches

def benchmark_model(model: nn.Module, dataloader: DataLoader, name: str) -> ModelStats:
    model.eval()
    model.to("cpu")

    dummy_input = next(iter(dataloader))[0][0].unsqueeze(0)

    macs    = get_model_macs(model, dummy_input)
    params  = get_num_parameters(model)
    latency = measure_latency(model, dummy_input)

    model.to(device)
    accuracy = evaluate(model, dataloader)

    return ModelStats(macs, params, latency, accuracy, name)

def compare_models(
    models_stats: list[ModelStats], 
    show_macs:    bool = True, 
    show_params:  bool = True,
    fig_size:     tuple = (10, 4)
) -> None:
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
    print(f"Name:     {model_stats.name}")
    print(f"Accuracy: {model_stats.accuracy:.2f}%")
    print(f"Latency:  {round(model_stats.latency * 1000, 1)} ms")
    print(f"Params:   {round(model_stats.params / 1e6)} M")
    print(f"MACs:     {round(model_stats.macs / 1e6)} M")

def get_dataset_size(image_size: int, channels: int, num_images: int, data_width: int) -> float:
    pixels = image_size * image_size * channels
    bits   = pixels * data_width
    return (bits / GiB) * num_images

def compare_single_values(
    values: list[float], 
    labels: list[str], 
    axis:   str = None, 
    title:  str = None
) -> None:
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(values))

    plt.barh(labels[::-1], values[::-1], color=colors)

    if axis: 
        plt.xlabel(axis)
    if title:
        plt.title(title)

    plt.show()

def compare_list_values(
    values:  dict[str, list],
    x_axis:  str = None,
    y_axis:  str = None,
    title:   str = None,
    y_range: tuple = None
) -> None:
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