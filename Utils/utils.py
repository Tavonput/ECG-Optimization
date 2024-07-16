# SHOULD BE DEPRECATED BUT WILL KEEP AROUND FOR A BIT
#
# import time
# import json
# import os
# import copy
# import logging
# import ast
# from copy import deepcopy
# from dataclasses import dataclass, asdict, field
# from collections import namedtuple, defaultdict
# from typing import Callable

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.optim import *
# from torch.optim.lr_scheduler import *

# from thop import profile

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm


# LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
# logging.basicConfig(format=LOG_FORMAT)
# logging.getLogger("UTL").setLevel(logging.INFO)
# log = logging.getLogger("UTL")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #===========================================================================================================================
# # Training And Eval
# #
# class TrainingStats:
#     """
#     TrainingStats stores training statistics and provides functions for serializing and deserializing the stats.

#     Parameters
#     ----------
#     from_save : str
#         A path to a serialized TrainingStats to load from.

#     Example Usage
#     -------------
#     ```
#     stats = TrainingStats()

#     # During training...
#     stats.best_accuracy = new_accuracy
#     stats.running_accuracy.append(new_accuracy)

#     # After training...
#     stats.training_time = training_time
#     stats.serialize("stats.txt")
#     stats.deserialize("stats.txt")
#     ```
#     """
#     def __init__(self, from_save: str = None) -> None:
#         self.best_accuracy    = 0.0
#         self.running_accuracy = []

#         self.running_train_time = []
#         self.running_epoch_time = []
#         self.total_train_time   = 0.0

#         self.epochs = 0

#         if from_save is not None:
#             self.deserialize(from_save)


#     def serialize(self, path: str) -> None:
#         path_dir = os.path.dirname(path)
#         if not os.path.exists(path_dir):
#             os.mkdir(path_dir)

#         with open(path, "w") as file:
#             file.write(f"{self.best_accuracy}\n")
#             file.write(f"{self.running_accuracy}\n")
#             file.write(f"{self.running_train_time}\n")
#             file.write(f"{self.running_epoch_time}\n")
#             file.write(f"{self.training_time}\n")
#             file.write(f"{self.epochs}")
    

#     def deserialize(self, path: str) -> None:
#         if not os.path.exists(path):
#             log.error(f"{path} does not exist")
#             return

#         with open(path, "r") as file:
#             lines = file.readlines()
#             self.best_accuracy      = float(lines[0].strip())
#             self.running_accuracy   = ast.literal_eval(lines[1].strip())
#             self.running_train_time = ast.literal_eval(lines[2].strip())
#             self.running_epoch_time = ast.literal_eval(lines[3].strip())
#             self.training_time      = float(lines[4].strip())
#             self.epochs             = int(lines[5].strip())


#     def display(self):
#         log.info("Displaying TrainingStats")
#         log.info(f"\tBest Accuracy:      {self.best_accuracy}")
#         log.info(f"\tRunning Accuracy:   {self.running_accuracy}")
#         log.info(f"\tRunning Train Time: {self.running_train_time}")
#         log.info(f"\tRunning Epoch Time: {self.running_epoch_time}")
#         log.info(f"\tTraining Time:      {self.training_time}")
#         log.info(f"\tEpochs:             {self.epochs}")


# def train(
#     model:      nn.Module,
#     dataloader: DataLoader,
#     criterion:  nn.Module,
#     optimizer:  Optimizer,
#     scheduler:  LambdaLR
# ) -> None:
#     """
#     Train a model for one epoch.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to train.
#     dataloader : Dataloader
#         Training dataloader.
#     criterion : nn.Module
#         Criterion.
#     optimizer : Optimizer
#         Optimizer.
#     scheduler : LambdaLR
#         Scheduler.
#     """
#     model.train()

#     for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         loss.backward()

#         optimizer.step()
#         scheduler.step()


# @torch.inference_mode()
# def evaluate(
#     model:      nn.Module,
#     dataloader: DataLoader,
#     verbose:    bool = True
# ) -> float:
#     """
#     Evaluate a model.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to evaluate.
#     dataloader : Dataloader
#         Testing dataloader.
#     verbose : bool
#         Verbosity.

#     Returns
#     -------
#     accuracy : float
#         The accuracy
#     """
#     model.eval()

#     num_samples = 0
#     num_correct = 0

#     for inputs, labels in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         outputs = outputs.argmax(dim=1)

#         num_samples += labels.size(0)
#         num_correct += (outputs == labels).sum()
    
#     return (num_correct / num_samples * 100).item()


# @torch.inference_mode()
# def evaluate_per_class(
#     model:       nn.Module,
#     dataloader:  DataLoader,
#     num_classes: int,
#     verbose:     bool = True
# ) -> dict:
#     """
#     Evaluate a model.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to evaluate.
#     dataloader : Dataloader
#         Testing dataloader.
#     num_classes: int
#         The number of classes.
#     verbose : bool
#         Verbosity.

#     Returns
#     -------
#     accuracy : float
#         The accuracy
#     """
#     model.eval()

#     total_samples_per_class   = torch.zeros(num_classes)
#     correct_samples_per_class = torch.zeros(num_classes)

#     for inputs, labels in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         outputs = outputs.argmax(dim=1)

#         for label in range(num_classes):
#             total_samples_per_class[label]   += (labels == label).sum().item()
#             correct_samples_per_class[label] += ((labels == label) & (outputs == label)).sum().item()

#     per_class_accuracy = (correct_samples_per_class / total_samples_per_class) * 100
#     per_class_accuracy_dict = {f"class_{i}": per_class_accuracy[i].item() for i in range(num_classes)}

#     return per_class_accuracy_dict


# def finetune(
#     model:       nn.Module,
#     epochs:      int,
#     dataloader:  dict[str, DataLoader],
#     save_path:   str,
#     lr:          float = 0.01,
#     safety:      int   = 0,
#     safety_dir:  str   = None,
#     do_eval:     bool  = True
# ) -> TrainingStats:
#     """
#     Basic finetune implementation.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to finetune.
#     epochs : int
#         Number of epochs to finetune.
#     dataloader : dict {str : DataLoader}
#         Both train and test loaders in dict {name, DataLoader}.
#     save_path : str
#         Where to save the model. 
#     lr : float
#         Initial learning rate for SGD.
#     safety : int
#         Epoch interval to save a checkpoint.
#     safety_dir : str
#         Save directory for safety checkpoints.
#     do_eval : bool
#         Whether or not to evaluate after every epoch.

#     Returns
#     -------
#     stats : TrainingStats
#         Statistics during training
#     """
#     if safety != 0 and not safety_dir:
#         log.error("No safety directory specified")
#         return
    
#     if safety != 0 and not os.path.exists(safety_dir):
#         os.makedirs(safety_dir)

#     save_dir = os.path.dirname(save_path)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     model.to(device)
#     optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, epochs)
#     criterion = nn.CrossEntropyLoss()

#     best_model_checkpoint = dict()
#     stats = TrainingStats()
#     stats.epochs = epochs

#     log.info("Begin finetuning...")
#     start_time = time.time()
#     for epoch in range(epochs):
#         epoch_start_time = time.time()

#         train_start_time = time.time()
#         train(model, dataloader["train"], criterion, optimizer, scheduler)
#         train_end_time = time.time()

#         stats.running_train_time.append(train_end_time - train_start_time)

#         if do_eval:
#             accuracy = evaluate(model, dataloader["test"])
#             stats.running_accuracy.append(accuracy)

#             if accuracy > stats.best_accuracy:
#                 best_model_checkpoint["state_dict"] = copy.deepcopy(model.state_dict())
#                 stats.best_accuracy = accuracy

#             epoch_end_time = time.time()
#             log.info(f"\tEpoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {stats.best_accuracy:.2f}%. Time: {(epoch_end_time - epoch_start_time)}s")
#         else:
#             epoch_end_time = time.time()
#             log.info(f"\tEpoch {epoch+1} Finished. Time: {(epoch_end_time - epoch_start_time)}s")

#         stats.running_epoch_time.append(epoch_end_time - epoch_start_time)

#         # Checkpoint safety        
#         if safety != 0 and (epoch + 1) % safety == 0:
#             checkpoint = {
#                 "model_state_dict":     model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "scheduler_state_dict": scheduler.state_dict(),
#                 "epoch":                epoch,
#             }
#             torch.save(checkpoint, f"{safety_dir}/check_{epoch+1}.pth")
#             log.info(f"\tSaved epoch {epoch+1} to {safety_dir}/check_{epoch+1}.pth")
        
#     end_time = time.time()
#     log.info("Finetuning completed")

#     stats.training_time = end_time - start_time

#     if do_eval:
#         torch.save(best_model_checkpoint["state_dict"], save_path)
#     else:
#         torch.save(model.state_dict(), save_path)

#     log.info(f"Model saved to {save_path}")
#     return stats


# def warm_up_dataloader(dataloader, num_batches: int = 0) -> None:
#     """
#     Warmup a dataloader by iterating over the batches and doing nothing. Can help with caching.

#     Parameters
#     ----------
#     dataloader : Dataloader
#         Dataloader to warmup.
#     num_batches : int
#         Number of batches to load. If not specified, all batches will be used.
#     """
#     if num_batches == 0:
#         for images, labels in tqdm(dataloader, desc="Warm-up", leave=False):
#             continue
#     else:
#         i = 1
#         for images, labels in enumerate(dataloader):
#             if i >= num_batches:
#                 break
#             i += 1


# #===========================================================================================================================
# # Benchmarking
# #
# Byte = 8
# KiB  = 1024 * Byte
# MiB  = 1024 * KiB
# GiB  = 1024 * MiB

# @dataclass
# class ModelStats:
#     name:     str   = "My Model"
#     macs:     int   = 0
#     params:   int   = 0
#     latency:  float = 0
#     accuracy: float = 0
    

# def add_model_stat_to_json(path: str, data: ModelStats) -> None:
#     """
#     Append a ModelStats instance to a json that stores a list of ModelStats.

#     Parameters
#     ----------
#     path : str
#         Path to json file.
#     data : ModelStats
#         ModelStats instance to add.
#     """
#     print(f"[I]: Adding model stat to {path}")
#     if not os.path.exists(path):
#         with open(path, "w") as _:
#             pass

#     with open(path, "r") as file:
#         try:
#             database = json.load(file)
#         except:
#             database = []

#     database.append(asdict(data))
#     with open(path, "w") as file:
#         json.dump(database, file, indent=4)


# def get_model_stats_from_json(path: str, model_names: list[str], sort: bool = False) -> list[ModelStats]:
#     """
#     Get a list of ModelStats instances from a json file by name searching.

#     Parameters
#     ----------
#     path : str
#         Path to json file.
#     model_names : list[str]
#         List of names corresponding to the ModelStats to retrieve.
#     sort : bool
#         Whether or not to sort the results based on the model_names.

#     Returns
#     -------
#     stats : list[ModelStats]
#         List of ModelStats.
#     """
#     if not os.path.exists(path):
#         print(f"Path does not exist {path}")
#         return None

#     with open(path, "r") as file:
#         try:
#             database = json.load(file)
#         except:
#             print(f"There is no data in {path}")
#             return None
        
#     # Get all model stats for the provided names
#     model_stats = filter(lambda x: x["name"] in model_names, database)
#     model_stats = map(lambda x: ModelStats(**x), model_stats)
#     model_stats = list(model_stats)

#     if sort:
#         sorted_stats = []
#         for model in model_names:
#             for stat in model_stats:
#                 if stat.name == model:
#                     sorted_stats.append(stat)
#                     break
#         model_stats = sorted_stats
    
#     return model_stats


# def get_model_macs(model: nn.Module, inputs: torch.Tensor) -> int:
#     """
#     Get model MACs through thop profile.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to profile.
#     inputs : torch.Tensor
#         Dummy input.

#     Returns
#     -------
#     macs : int
#         Number of MACs.
#     """
#     macs, _ = profile(model, inputs=(inputs,), verbose=False)
#     return macs


# def get_num_parameters(model: nn.Module) -> int:
#     """
#     Get the  number of parameters in a model.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to profile:

#     Returns
#     ------- 
#     count : int
#         Number of parameters.
#     """
#     num_counted_elements = 0
#     for param in model.parameters():
#         num_counted_elements += param.numel()
#     return num_counted_elements


# @torch.no_grad()
# def measure_latency(
#     model:       nn.Module, 
#     dummy_input: torch.Tensor, 
#     n_warmup:    int = 50, 
#     n_test:      int = 100,
#     test_device: str = "cpu",
# ) -> float:
#     """
#     Measure the latency of a model. Latency will be measured 10 times and averaged.

#     Parameters
#     ----------
#     model : nn.Module
#         Model to profile.
#     dummy_input : torch.Tensor
#         Dummy input that will run through the model.
#     n_warmup : int
#         Number of warmup iterations.
#     n_test : int
#         Number of forward passes.
#     test_device : str
#         Device to test on.

#     Returns
#     -------
#     latency : float
#         Latency in seconds.
#     """
#     model.eval()
#     model.to(test_device)
#     inp = dummy_input.to(test_device)

#     # Warmup
#     for _ in range(n_warmup):
#         _ = model(inp)

#     # Real test
#     times = []
#     for _ in range(0, 10):
#         t1 = time.time()
#         for _ in range(n_test):
#             _ = model(inp)
#         t2 = time.time()
#         times.append((t2 - t1) / n_test)
    
#     return sum(times) / len(times)


# def benchmark_dataloader(dataloader: DataLoader, num_batches: int = 100) -> float:
#     """
#     Test how long it takes to load a batch from a dataloader.

#     Parameters
#     ----------
#     dataloader : DataLoader
#         Dataloader to profile.
#     num_batches : int
#         Number of batches to test.

#     Returns
#     -------
#     time : float
#         Time to load one batch in seconds.
#     """
#     start_time = time.time()
#     for i, (images, labels) in enumerate(dataloader):
#         if i >= num_batches - 1:
#             break

#     end_time = time.time()
#     return (end_time - start_time) / num_batches


# def benchmark_model(model: nn.Module, dataloader: DataLoader, name: str, no_latency: bool = False) -> ModelStats:
#     """
#     Benchmark a model for accuracy, latency, parameter count, and MACs.

#     Parameters
#     ----------
#     model : nn.Module 
#         Model to benchmark.
#     dataloader : DataLoader
#         Testing dataloader.
#     name : str
#         Name to be attached with the results.
#     no_latency : bool
#         Whether or not to skip latency profile.

#     Return
#     ------
#     stats : ModelStats
#         ModelStats instance.
#     """
#     print(f"[I]: Benchmarking model {name}")
#     model.eval()
#     model.to("cpu")

#     dummy_input = next(iter(dataloader))[0][0].unsqueeze(0)

#     print(f"[I]: \tGetting model params and MACs")
#     macs    = get_model_macs(deepcopy(model).to(torch.float32), dummy_input.clone().to(torch.float32))
#     params  = get_num_parameters(model)

#     if no_latency:
#         print(f"[I]: \tLatency measurement skipped")
#         latency = 0
#     else:
#         print(f"[I]: \tMeasuring latency")
#         latency = measure_latency(model, dummy_input)

#     print(f"[I]: \tEvaluating")
#     model.to(device)
#     accuracy = evaluate(model, dataloader)

#     print(f"[I]: Benchmarking for {name} finished")
#     return ModelStats(
#         macs     = macs, 
#         params   = params, 
#         latency  = latency, 
#         accuracy = accuracy, 
#         name     = name
#     )


# def get_dataset_size(image_size: int, channels: int, num_images: int, data_width: int) -> float:
#     """
#     Get the size of a dataset in GiB. Images must be square.

#     Parameters
#     ----------
#     image_size : int
#         Image resolution.
#     channels : int
#         Number of channels.
#     num_images : int
#         Number of images.
#     data_width : int
#         Bit size of each value.

#     Returns
#     -------
#     size : float
#         Dataset size in GiB.
#     """
#     pixels = image_size * image_size * channels
#     bits   = pixels * data_width
#     return (bits / GiB) * num_images


# def compute_top_mean_std(values: np.ndarray, discard: int) -> tuple[float, float, float]:
#     """
#     Compute the top value, mean, and std of a given array.

#     Parameters
#     ----------
#     values : np.ndarray
#         The array of values.
#     discard : int
#         How many initial values to discard for mean and std.

#     Returns
#     -------
#     stats : float, float, float
#         The top, mean, and std.
#     """
#     top  = max(values)
#     mean = np.mean(values[discard:])
#     std  = np.std(values[discard:])
#     return top, mean, std


# def time_to_convergence(accuracies: list[float], time_per_epoch: float, std_threshold: float, window_size: int) -> tuple[int, float]:
#     """
#     Compute the time it takes for an array of accuracies to converge. Convergence is defined as...
#         std(last_window_size_accuracies) <= std_threshold
        
#     For example, if the window size is 5, then convergence is when the std of the last 5 runs is less than or equal to the
#     std threshold.

#     This function will return 0s if convergence was not met.

#     Parameters
#     ----------
#     accuracies : list[float]
#         The running accuracies.
#     time_per_epoch : float
#         The amount of time per epoch.
#     std_threshold : float
#         The std to compare against the window std.
#     window_size : int
#         How many previous entries should be considered during convergence testing.
    
#     Returns
#     -------
#     epoch, time : int, float
#         The epoch of convergence and the time to converge.
#     """
#     current_idx  = 0
#     epoch        = 0
#     while current_idx + window_size <= len(accuracies):
#         window_std = np.std(accuracies[current_idx : (current_idx + window_size)])

#         if window_std <= std_threshold:
#             epoch = current_idx + window_size
#             break

#         current_idx += 1
    
#     if epoch == 0:
#         log.info(f"Accuracies did not converge")
#         return 0, 0
    
#     return epoch, epoch * time_per_epoch


# #===========================================================================================================================
# # Display and Plotting
# #
# # TODO: Currently in the process of refactoring. Some function need to use the PlotConfig struct. Use subplot ax instead
# #       of plt.
# #
# @dataclass
# class PlotConfig:
#     title:   str  = "Title"
#     legend:  list = field(default_factory=list)

#     x_label: str = "x-axis"
#     y_label: str = "y-axis"

#     x_range: tuple = None
#     y_range: tuple = None

#     x_scale: str = "linear"
#     y_scale: str = "linear"

#     x_grid: bool = True
#     y_grid: bool = True

#     fig_size: tuple = (10, 4)


# class GroupedValues:
#     """
#     Helper class for grouping values together. For example, this class is used for grouped bar plots.

#     Internally groups are stored as a dictionary with the key being the name of a group and the value being the group.
#     A group is a list of Entry tuples where an Entry has a value and error.
#     """
#     Entry = namedtuple("Entry", ["value", "error"])

#     def __init__(self) -> None:
#         self.groups = defaultdict(list)


#     def is_empty(self) -> bool:
#         if len(self.groups) == 0:
#             return True
#         return False


#     def is_valid(self) -> bool:
#         if self.is_empty():
#             return False
        
#         # The length of the first group
#         baseline = len(list(self.groups.values())[0])

#         valid = True
#         for group in self.groups.values():
#             valid &= (len(group) == baseline)

#         return valid
    

#     def size(self) -> int:
#         return len(self.groups)


#     def group_size(self) -> int:
#         if self.is_empty():
#             return 0
        
#         if self.is_valid() is False:
#             print("Group sizes are not all the same")

#         # The length of the first group
#         return len(list(self.groups.values())[0])


#     def add_to_group(self, group: str, value: int|float, error: int|float = 0) -> None:
#         self.groups[group].append(GroupedValues.Entry(value, error))


# def compare_models(
#     models_stats: list[ModelStats], 
#     show_macs:    bool = True, 
#     show_params:  bool = True,
#     fig_size:     tuple = (10, 4)
# ) -> None:
#     """
#     Compare a list of ModelStats displayed with a bar graphs.

#     @param model_stats: List of ModelStats to compare.
#     @param show_macs: Whether or not to show the MACs plot.
#     @param show_params: Whether or not to show the params plot.
#     @param fig_size: Figure size of the plots.
#     """
#     sns.set_style("whitegrid")

#     names   = [model.name for model in models_stats]
#     accs    = [model.accuracy for model in models_stats]
#     macs    = [round(model.macs / 1e6) for model in models_stats]
#     latency = [round(model.latency * 1000, 1) for model in models_stats]
#     params  = [round(model.params / 1e6) for model in models_stats]

#     plots = 2
#     if show_macs: 
#         plots += 1
#     if show_params: 
#         plots += 1

#     fig, axs = plt.subplots(1, plots, figsize=fig_size)
#     colors = sns.color_palette("husl", len(names))

#     axs[0].bar(names, accs, color=colors)
#     axs[0].set_title("Accuracy")
#     acc_min_bound = np.clip(min(accs) - 10, 0, 100)
#     acc_max_bound = np.clip(max(accs) + 5, 0, 100)
#     axs[0].set_ylim([acc_min_bound, acc_max_bound])

#     axs[1].bar(names, latency, color=colors)
#     axs[1].set_title("Latency (ms)")

#     if show_params:
#         axs[2].bar(names, params, color=colors)
#         axs[2].set_title("Parameters (M)")

#     if show_macs:
#         axs[3].bar(names, macs, color=colors)
#         axs[3].set_title("MACs (M)")

#     plt.tight_layout()
#     plt.show()


# def display_model_stats(model_stats: ModelStats) -> None:
#     """
#     Formatted display of a ModelStats instance.

#     @param model_stats: ModelStats instance.
#     """
#     print(f"Name:     {model_stats.name}")
#     print(f"Accuracy: {model_stats.accuracy:.2f}%")
#     print(f"Latency:  {round(model_stats.latency * 1000, 1)} ms")
#     print(f"Params:   {round(model_stats.params / 1e6)} M")
#     print(f"MACs:     {round(model_stats.macs / 1e6)} M")


# def bar_plot(
#     values:     dict[str, float|int],
#     config:     PlotConfig,
#     horizontal: bool = True,
# ) -> None:
#     """
#     Bar graph.

#     Parameters
#     ----------
#     values : dict[str, float|int]
#         Dictionary of {names : values}.
#     config : PlotConfig
#         The plotting config.
#     horizontal : bool
#         Whether or not to have a horizontal plot.
#     """
#     sns.set_style("whitegrid")
#     colors = sns.color_palette("husl", len(values))

#     labels = list(values.keys())
#     values = list(values.values())

#     _, ax = plt.subplots(figsize=config.fig_size)

#     if horizontal:
#         bars = ax.barh(labels[::-1], values[::-1], color=colors)
#         ax.set_xlabel(config.x_label)
#         ax.set_xscale(config.x_scale)
#         ax.set_xlim(config.x_range)

#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(
#                 f"{height}", 
#                 xy = (bar.get_y() + bar.get_width() / 2, height),
#                 xytext = (3, 0),
#                 textcoords = "offset points",
#                 ha = "center", va = "bottom", 
#             )
#     else:
#         bars = ax.bar(labels, values, color=colors)
#         ax.set_ylabel(config.y_label)
#         ax.set_yscale(config.y_scale)
#         ax.set_ylim(config.y_range)

#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(
#                 f"{height}", 
#                 xy = (bar.get_x() + bar.get_width() / 2, height),
#                 xytext = (0, 3),
#                 textcoords = "offset points",
#                 ha = "center", va = "bottom", 
#             )

#     ax.set_title(config.title)
#     ax.grid(config.x_grid, axis="x")
#     ax.grid(config.y_grid, axis="y")

#     plt.show()


# def bar_plot_error(
#     values:     dict[str, tuple[float|int, float]],
#     config:     PlotConfig,
#     horizontal: bool = True,
# ) -> None:
#     """
#     Bar graph with error bars.

#     Parameters
#     ----------
#     values : dict[str, tuple[float|int, float]]
#         Dictionary of {name : (value, error)}.
#     config : PlotConfig
#         The plotting config.
#     horizontal : bool
#         Whether or not to have a horizontal plot.
#     """
#     sns.set_style("whitegrid")
#     colors = sns.color_palette("husl", len(values))

#     labels  = list(values.keys())
#     numbers = [v[0] for v in values.values()]
#     errors  = [v[1] for v in values.values()]

#     _, ax = plt.subplots(figsize=config.fig_size)

#     if horizontal:
#         ax.barh(labels[::-1], numbers[::-1], yerr=errors[::-1], color=colors)
#         ax.set_xscale(config.x_scale)
#         ax.set_xlim(config.x_range)
#     else:
#         ax.bar(labels, numbers, yerr=errors, color=colors)
#         ax.set_yscale(config.y_scale)
#         ax.set_ylim(config.y_range)

#     ax.set_ylabel(config.y_label)
#     ax.set_xlabel(config.x_label)
#     ax.set_title(config.title)
#     ax.grid(config.x_grid, axis="x")
#     ax.grid(config.y_grid, axis="y")

#     plt.show()


# def compare_list_values(
#     values:  dict[str, list],
#     x_axis:  str   = None,
#     y_axis:  str   = None,
#     title:   str   = None,
#     y_range: tuple = None
# ) -> None:
#     """
#     Compare lists of values through a plot.

#     @param values: Dictionary containing {"label": list of values}.
#     @param x_axis: Label for x-axis.
#     @param y_axis: Label for y_axis.
#     @param title: Title.
#     @param r_range: Range of display for the y-axis (min, max).
#     """
#     sns.set_style("whitegrid")
#     sns.set_palette("husl")

#     for k, v in values.items():
#         plt.plot(v, label=k)

#     if y_range:
#         plt.ylim(y_range[0], y_range[1])
#     if x_axis:
#         plt.xlabel(x_axis)
#     if y_axis:
#         plt.ylabel(y_axis)
#     if title:
#         plt.title(title)
        
#     plt.legend()
#     plt.show()


# def compare_pairwise(
#     values: dict[str, list[tuple]],
#     config: PlotConfig
# ) -> None:
#     """
#     Normal pairwise (x, y) plot.

#     @param values: Dictionary containing labels and points {name : list of (x, y)}.
#     @param config: PlotConfig.
#     """
#     sns.set_style("whitegrid")
#     sns.set_palette("husl")
#     _, ax = plt.subplots(figsize=config.fig_size)

#     for k, v in values.items():
#         xs, ys = zip(*v)
#         ax.plot(xs, ys, "o-", label=k)

#     if config.y_range:
#         ax.set_ylim(config.y_range[0], config.y_range[1])    
#     if config.x_range:
#         ax.set_ylim(config.x_range[0], config.x_range[1])   

#     ax.set_xlabel(config.x_label)
#     ax.set_ylabel(config.y_label)
#     ax.set_title(config.title)
#     ax.legend()
#     plt.show()


# def visualize_image_from_numpy(image: np.ndarray, transpose: bool = False, single_channel: bool = False) -> None:
#     """
#     Simple visualization for an image from a numpy array.

#     Parameters
#     ----------
#     image : np.ndarray
#         The image in the form of a numpy array.
#     transpose : bool 
#         Change image from (c, h, w) to (h, w, c).
#     single_channel : bool
#         Show each channel independently.
#     """
#     if single_channel:
#         _, ax = plt.subplots(1, 3, figsize=(12, 4))
#         for i, img in enumerate(image):
#             ax[i].imshow(img)
#             ax[i].axis(False)
#     else:
#         if transpose:
#             image = image.transpose(1, 2, 0)

#         _, ax = plt.subplots()
#         ax.imshow(image)
#         ax.axis(False)

#     plt.show()


# def grouped_bar_plot(groups: GroupedValues, config: PlotConfig, callback: Callable = None, **kwargs) -> None:
#     """
#     Grouped bar plot. Has only been tested with a grouping size of 2.

#     Parameters
#     ----------
#     groups : GroupedValues
#         The groups (see GroupedValues on how to set this up).
#     config : PlotConfig
#         The plotting config.
#     callback : Callable
#         A callback function for user defined plotting. function(ax, config, **kwargs).
#     **kwargs
#         Passed to the callback function.
#     """
#     _, ax  = plt.subplots(figsize=config.fig_size)
#     colors = sns.color_palette("husl", groups.group_size())

#     # Set the location of the x-ticks based of the group sizes.
#     start_tick = (groups.group_size() - 1) / 2
#     interval   = groups.group_size() + 1
#     x_ticks    = [(start_tick + (i * interval)) for i in range(groups.size())]
#     ax.set_xticks(x_ticks)
    
#     labels = []
#     current_bar = 0
#     for name, group in groups.groups.items():
#         for i, entry in enumerate(group):
#             if entry.error != 0:
#                 ax.bar(current_bar, entry.value, yerr=entry.error, color=colors[i])
#             else:
#                 ax.bar(current_bar, entry.value, color=colors[i])
#             current_bar += 1

#         current_bar += 1
#         labels.append(name)

#     ax.set_xticklabels(labels)

#     ax.set_ylim(config.y_range)
#     ax.grid(False, axis="x")
#     ax.set_title(config.title)
#     ax.set_xlabel(config.x_label)
#     ax.legend(config.legend)

#     if callback:
#         callback(ax, config, **kwargs)

#     plt.show()


# def display_training_stat_table(stats: list[TrainingStats], base: TrainingStats, names: list[str], title: str, base_discard: int = 20) -> None:
#     """
#     Display a table of different training stats compared against a baseline.

#     Parameters
#     ----------
#     stats : list[TrainingStats]
#         The list of training stats.
#     base : TrainingStats
#         The baseline training stats.
#     names : list[str]
#         The list of names for each training stats.
#     title : str
#         The title of the table.
#     base_discard : int
#         The epoch discard for the baseline.
#     """
#     print(f"=== {title} | Top   | Mean  | STD    | Abs    | Rel")

#     top_base, mean_base, std_base = compute_top_mean_std(base.running_accuracy, discard=base_discard)
#     print(f"Base    | {round(top_base, 2):.2f} | {round(mean_base, 2):.2f} | {round(std_base, 4):.4f} | ------ | ------")

#     for i, stat in enumerate(stats):
#         top, mean, std = compute_top_mean_std(stat.running_accuracy, discard=10)
#         mean_diff      = mean - mean_base
#         mean_diff_norm = mean / mean_base

#         print(f"{names[i]}| {round(top, 2):.2f} | {round(mean, 2):.2f} | {round(std, 4):.4f} | {round(mean_diff, 4):.4f} | {round(mean_diff_norm, 4):.4f}")


# #===========================================================================================================================
# # Main (Used for testing this file)
# #
# if __name__ == "__main__":
#     """
#     Main for testing.
#     """
#     group = GroupedValues()

#     group.add_to_group("group_1", 1, 1)
#     group.add_to_group("group_2", 1, 1)
#     print(group.is_valid())