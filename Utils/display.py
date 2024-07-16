from dataclasses import dataclass, field
from collections import namedtuple, defaultdict
from typing import Callable

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .benchmarking import ModelStats, compute_top_mean_std
from .classification import TrainingStats


#===========================================================================================================================
# Display and Plotting
#
# TODO: Currently in the process of refactoring. Some function need to use the PlotConfig struct. Use subplot ax instead
#       of plt.
#
@dataclass
class PlotConfig:
    title:   str  = "Title"
    legend:  list = field(default_factory=list)

    x_label: str = "x-axis"
    y_label: str = "y-axis"

    x_range: tuple = None
    y_range: tuple = None

    x_scale: str = "linear"
    y_scale: str = "linear"

    x_grid: bool = True
    y_grid: bool = True

    fig_size: tuple = (10, 4)


class GroupedValues:
    """
    Helper class for grouping values together. For example, this class is used for grouped bar plots.

    Internally groups are stored as a dictionary with the key being the name of a group and the value being the group.
    A group is a list of Entry tuples where an Entry has a value and error.
    """
    Entry = namedtuple("Entry", ["value", "error"])

    def __init__(self) -> None:
        self.groups = defaultdict(list)


    def is_empty(self) -> bool:
        if len(self.groups) == 0:
            return True
        return False


    def is_valid(self) -> bool:
        if self.is_empty():
            return False
        
        # The length of the first group
        baseline = len(list(self.groups.values())[0])

        valid = True
        for group in self.groups.values():
            valid &= (len(group) == baseline)

        return valid
    

    def size(self) -> int:
        return len(self.groups)


    def group_size(self) -> int:
        if self.is_empty():
            return 0
        
        if self.is_valid() is False:
            print("Group sizes are not all the same")

        # The length of the first group
        return len(list(self.groups.values())[0])


    def add_to_group(self, group: str, value: int|float, error: int|float = 0) -> None:
        self.groups[group].append(GroupedValues.Entry(value, error))


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


def bar_plot(
    values:     dict[str, float|int],
    config:     PlotConfig,
    horizontal: bool = True,
) -> None:
    """
    Bar graph.

    Parameters
    ----------
    values : dict[str, float|int]
        Dictionary of {names : values}.
    config : PlotConfig
        The plotting config.
    horizontal : bool
        Whether or not to have a horizontal plot.
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


def bar_plot_error(
    values:     dict[str, tuple[float|int, float]],
    config:     PlotConfig,
    horizontal: bool = True,
) -> None:
    """
    Bar graph with error bars.

    Parameters
    ----------
    values : dict[str, tuple[float|int, float]]
        Dictionary of {name : (value, error)}.
    config : PlotConfig
        The plotting config.
    horizontal : bool
        Whether or not to have a horizontal plot.
    """
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(values))

    labels  = list(values.keys())
    numbers = [v[0] for v in values.values()]
    errors  = [v[1] for v in values.values()]

    _, ax = plt.subplots(figsize=config.fig_size)

    if horizontal:
        ax.barh(labels[::-1], numbers[::-1], yerr=errors[::-1], color=colors)
        ax.set_xscale(config.x_scale)
        ax.set_xlim(config.x_range)
    else:
        ax.bar(labels, numbers, yerr=errors, color=colors)
        ax.set_yscale(config.y_scale)
        ax.set_ylim(config.y_range)

    ax.set_ylabel(config.y_label)
    ax.set_xlabel(config.x_label)
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


def visualize_image_from_numpy(image: np.ndarray, transpose: bool = False, single_channel: bool = False) -> None:
    """
    Simple visualization for an image from a numpy array.

    Parameters
    ----------
    image : np.ndarray
        The image in the form of a numpy array.
    transpose : bool 
        Change image from (c, h, w) to (h, w, c).
    single_channel : bool
        Show each channel independently.
    """
    if single_channel:
        _, ax = plt.subplots(1, 3, figsize=(12, 4))
        for i, img in enumerate(image):
            ax[i].imshow(img)
            ax[i].axis(False)
    else:
        if transpose:
            image = image.transpose(1, 2, 0)

        _, ax = plt.subplots()
        ax.imshow(image)
        ax.axis(False)

    plt.show()


def grouped_bar_plot(groups: GroupedValues, config: PlotConfig, callback: Callable = None, **kwargs) -> None:
    """
    Grouped bar plot. Has only been tested with a grouping size of 2.

    Parameters
    ----------
    groups : GroupedValues
        The groups (see GroupedValues on how to set this up).
    config : PlotConfig
        The plotting config.
    callback : Callable
        A callback function for user defined plotting. function(ax, config, **kwargs).
    **kwargs
        Passed to the callback function.
    """
    _, ax  = plt.subplots(figsize=config.fig_size)
    colors = sns.color_palette("husl", groups.group_size())

    # Set the location of the x-ticks based of the group sizes.
    start_tick = (groups.group_size() - 1) / 2
    interval   = groups.group_size() + 1
    x_ticks    = [(start_tick + (i * interval)) for i in range(groups.size())]
    ax.set_xticks(x_ticks)
    
    labels = []
    current_bar = 0
    for name, group in groups.groups.items():
        for i, entry in enumerate(group):
            if entry.error != 0:
                ax.bar(current_bar, entry.value, yerr=entry.error, color=colors[i])
            else:
                ax.bar(current_bar, entry.value, color=colors[i])
            current_bar += 1

        current_bar += 1
        labels.append(name)

    ax.set_xticklabels(labels)

    ax.set_ylim(config.y_range)
    ax.grid(False, axis="x")
    ax.set_title(config.title)
    ax.set_xlabel(config.x_label)
    ax.legend(config.legend)

    if callback:
        callback(ax, config, **kwargs)

    plt.show()


def display_training_stat_table(stats: list[TrainingStats], base: TrainingStats, names: list[str], title: str, base_discard: int = 20) -> None:
    """
    Display a table of different training stats compared against a baseline.

    Parameters
    ----------
    stats : list[TrainingStats]
        The list of training stats.
    base : TrainingStats
        The baseline training stats.
    names : list[str]
        The list of names for each training stats.
    title : str
        The title of the table.
    base_discard : int
        The epoch discard for the baseline.
    """
    print(f"=== {title} | Top   | Mean  | STD    | Abs    | Rel")

    top_base, mean_base, std_base = compute_top_mean_std(base.running_accuracy, discard=base_discard)
    print(f"Base    | {round(top_base, 2):.2f} | {round(mean_base, 2):.2f} | {round(std_base, 4):.4f} | ------ | ------")

    for i, stat in enumerate(stats):
        top, mean, std = compute_top_mean_std(stat.running_accuracy, discard=10)
        mean_diff      = mean - mean_base
        mean_diff_norm = mean / mean_base

        print(f"{names[i]}| {round(top, 2):.2f} | {round(mean, 2):.2f} | {round(std, 4):.4f} | {round(mean_diff, 4):.4f} | {round(mean_diff_norm, 4):.4f}")
