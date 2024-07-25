import sys
sys.path.append("../../")

import os
import logging
import argparse
import yaml
from typing import Any
from collections import namedtuple
from copy import deepcopy

import torch

import Quantization.explicit_quant as quant

from Dataset.data_generation import ArrhythmiaLabels
from Dataset.dataset import build_dataloader

from Utils.model_loading import *
from Utils.classification import warm_up_dataloader, finetune, evaluate


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("OPTIM").setLevel(logging.INFO)
log = logging.getLogger("OPTIM")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_model(
    model_name:    str, 
    epochs:        int, 
    image_size:    int, 
    batch_size:    int, 
    train_path:    str,
    test_path:     str,
    save_dir_root: str,
    preload_train: bool,
    preload_test:  bool
) -> None:
    dataloader = build_dataloader(
        train_path    = train_path,
        test_path     = test_path,
        transform     = None,
        batch_size    = batch_size,
        preload_train = preload_train,
        preload_test  = preload_test,
    )

    model = load_model(model_name)
    replace_classifier(model_name, model, ArrhythmiaLabels.size)

    save_name = f"{model_name}_ep{epochs}_i{image_size}"
    save_dir  = f"{save_dir_root}/{save_name}"

    warm_up_dataloader(dataloader["train"])
    warm_up_dataloader(dataloader["test"])

    training_stats = finetune(
        model      = model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth"
    )

    training_stats.serialize(f"{save_dir}/stats.txt")


def window_shrinking(
    model_name:    str, 
    save_dir_root: str,
    resume:        str = None
) -> None:

    if resume is None:
        model = load_model(model_name)
        replace_classifier(model_name, model, ArrhythmiaLabels.size)
    else:
        model = load_model_from_pretrained(model_name, resume, ArrhythmiaLabels.size)

    TrainConfig  = namedtuple("TrainConfig", ["window_size", "ratio", "epochs", "batch_size"])
    train_config = [
        TrainConfig(256, 0.3, 10, 64),
        TrainConfig(248, 0.3, 10, 64),
        TrainConfig(240, 0.3, 10, 64),
        TrainConfig(232, 0.3, 10, 64),
        TrainConfig(224, 0.3, 10, 64),
        TrainConfig(216, 0.4, 10, 64),
        TrainConfig(208, 0.4, 10, 64),
        TrainConfig(200, 0.4, 10, 64),
        TrainConfig(192, 0.4, 10, 64),
        TrainConfig(184, 0.4, 10, 64),
        TrainConfig(176, 0.5, 10, 128),
        TrainConfig(168, 0.5, 10, 128),
        TrainConfig(160, 0.5, 10, 128),
        TrainConfig(152, 0.6, 10, 128),
        TrainConfig(144, 0.6, 10, 128),
        TrainConfig(136, 0.6, 10, 128),
        TrainConfig(128, 1.0, 30, 128),
    ]
    method = "balance"

    for config in train_config:
        print(f"===== Window Size {config.window_size} =====")

        if config.window_size == 128:
            train_path = f"../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_unfiltered_i{config.window_size}_train.h5"
        else:
            train_path = f"../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_unfiltered_i{config.window_size}_train_r{config.ratio}_{method}.h5"

        dataloader = build_dataloader(
            train_path    = train_path,
            test_path     = f"../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_unfiltered_i128_test.h5",
            transform     = None,
            batch_size    = config.batch_size,
            preload_train = True,
        )

        save_name = f"{model_name}_ep{config.epochs}_i{config.window_size}_r{config.ratio}"
        save_dir  = f"{save_dir_root}/{save_name}"

        warm_up_dataloader(dataloader["train"])
        if config.window_size == 128:
            warm_up_dataloader(dataloader["test"])

        training_stats = finetune(
            model      = model,
            epochs     = config.epochs,
            dataloader = dataloader,
            save_path  = f"{save_dir}/{save_name}.pth",
            do_eval    = (config.window_size == 128)
        )

        training_stats.serialize(f"{save_dir}/stats.txt")

        del dataloader


def pruning(
    model_name:      str,
    base_checkpoint: str,
    image_size:      int,
    batch_size:      int,
    epochs:          int,
    train_path:      str,
    test_path:       str,
    save_dir_root:   str,
    preload_train:   bool,
    preload_test:    bool,
    prune_ratio:     float,
    global_prune:    bool,
) -> None:
    base_model = load_model_from_pretrained(model_name, base_checkpoint, ArrhythmiaLabels.size)

    pruned_model = deepcopy(base_model).to("cpu")
    pruned_model.eval()

    dummy_input = torch.rand((1, 3, image_size, image_size))

    classifier = get_classifier(model_name, pruned_model)

    imp    = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MagnitudePruner(
        model             = pruned_model,
        example_inputs    = dummy_input,
        importance        = imp,
        global_pruning    = global_prune,
        pruning_ratio     = prune_ratio,
        ignored_layers    = [classifier],
    )
    pruner.step()

    dataloader = build_dataloader(
        train_path    = train_path,
        test_path     = test_path,
        transform     = None,
        batch_size    = batch_size,
        preload_train = preload_train,
        preload_test  = preload_test,
    )
    warm_up_dataloader(dataloader["train"])
    warm_up_dataloader(dataloader["test"])

    if global_prune:
        save_name = f"{model_name}_ep{epochs}_i{image_size}_p{prune_ratio}_global"
    else:
        save_name = f"{model_name}_ep{epochs}_i{image_size}_p{prune_ratio}_layer"
    save_dir = f"{save_dir_root}/{save_name}"

    training_stats = finetune(
        model      = pruned_model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth",
        full_save  = True,
    )
    training_stats.serialize(f"{save_dir}/stats.txt")


def quantize(
    model_name:    str, 
    model_path:    str,
    full_load:     bool,
    prune_ratio:   float,
    bits:          int,
    epochs:        int, 
    image_size:    int, 
    batch_size:    int, 
    train_path:    str,
    test_path:     str,
    save_dir_root: str,
    preload_train: bool,
    preload_test:  bool
):
    quant.quant_initialize()
    quant.set_weight_calibrators("max", bits)

    # MODEL LOADING IMPLEMENTATION IS TEMPORARY 
    model = load_model_from_pretrained(model_name, model_path, ArrhythmiaLabels.size, full_load)
    os.makedirs("tmp")
    torch.save(model.state_dict(), "tmp/state_dict.pth")
    model = load_from_layer_pruned(model_name, "tmp/state_dict.pth", prune_ratio, torch.rand((1, 3, 128, 128)))
    os.remove("tmp/state_dict.pth")
    os.removedirs("tmp")
    print(model)
    model.to(device)

    dataloader = build_dataloader(
        train_path    = train_path,
        test_path     = test_path,
        batch_size    = batch_size,
        transform     = None,
        preload_train = preload_train,
        preload_test  = preload_test,
    )

    save_name = f"{model_name}_ep{epochs}_i{image_size}"
    save_dir  = f"{save_dir_root}/{save_name}"

    quant.calibrate(model=model, dataloader=dataloader["train"], num_batches=100, method="max")

    if bits != 8:
        log.info("Cannot export to ONNX because bits != 8")
    else:
        quant.export_to_onnx(model, image_size, 1, f"{save_dir_root}/{save_name}_ptq.onnx")

    training_stats = finetune(
        model      = model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth",
        full_save  = full_load,
    )
    training_stats.serialize(f"{save_dir}/stats.txt")

    model = load_model_from_pretrained(model_name, f"{save_dir}/{save_name}.pth", ArrhythmiaLabels.size, full_load)
    model.to(device)

    if bits != 8:
        log.info("Cannot export to ONNX because bits != 8")
    else:
        quant.export_to_onnx(model, image_size, 1, f"{save_dir_root}/{save_name}_qat.onnx")


#===========================================================================================================================
# Main
#
def load_config(file_path: str) -> Any:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type    = str, 
        default = None, 
        help    = "Path to config yml file"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    config = load_config(args.config)

    if config["base_model"]["do_base"]:
        base_model(
            model_name    = config["base_model"]["model_name"],
            epochs        = config["base_model"]["epochs"],
            image_size    = config["base_model"]["image_size"],
            batch_size    = config["base_model"]["batch_size"],
            train_path    = config["base_model"]["train_path"],
            test_path     = config["base_model"]["test_path"],
            save_dir_root = config["base_model"]["save_dir"],
            preload_train = config["base_model"]["preload_train"],
            preload_test  = config["base_model"]["preload_test"],
        )

    if config["shrinking"]["do_shrink"]:
        if config["shrinking"]["resume"] is True:
            resume = config["shrinking"]["resume_checkpoint"]
        else:
            resume = None

        window_shrinking(
            model_name    = config["shrinking"]["model_name"],
            save_dir_root = config["shrinking"]["save_dir"],
            resume        = resume,
        )

    if config["pruning"]["do_prune"]:
        pruning(
            model_name      = config["pruning"]["model_name"],
            base_checkpoint = config["pruning"]["base_checkpoint"],
            image_size      = config["pruning"]["image_size"],
            batch_size      = config["pruning"]["batch_size"],
            epochs          = config["pruning"]["epochs"],
            train_path      = config["pruning"]["train_path"],
            test_path       = config["pruning"]["test_path"],
            save_dir_root   = config["pruning"]["save_dir"],
            preload_train   = config["pruning"]["preload_train"],
            preload_test    = config["pruning"]["preload_test"],
            prune_ratio     = config["pruning"]["prune_ratio"],
            global_prune    = config["pruning"]["global_prune"],
        )

    if config["quantization"]["do_quant"]:
        quantize(    
            model_name    = config["quantization"]["model_name"],
            model_path    = config["quantization"]["model_path"],
            full_load     = config["quantization"]["from_pruned"],
            prune_ratio   = config["quantization"]["prune_ratio"],
            bits          = config["quantization"]["bits"],
            epochs        = config["quantization"]["epochs"],
            image_size    = config["quantization"]["image_size"],
            batch_size    = config["quantization"]["batch_size"],
            train_path    = config["quantization"]["train_path"],
            test_path     = config["quantization"]["test_path"],
            save_dir_root = config["quantization"]["save_dir"],
            preload_train = config["quantization"]["preload_train"],
            preload_test  = config["quantization"]["preload_test"],
        )


if __name__ == "__main__":
    main()