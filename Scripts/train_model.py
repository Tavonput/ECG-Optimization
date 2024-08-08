import sys
sys.path.append("../")

import logging
import argparse
import yaml
from typing import Any

import torch

from Dataset.data_generation import ArrhythmiaLabels
from Dataset.dataset import build_dataloader

from Utils.model_loading import *
from Utils.classification import warm_up_dataloader, finetune
from Utils.distributed import *



LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("TRAIN").setLevel(logging.INFO)
log = logging.getLogger("TRAIN")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(
    model_name:    str, 
    epochs:        int, 
    image_size:    int, 
    batch_size:    int, 
    train_path:    str,
    test_path:     str,
    save_dir_root: str,
    preload_train: bool,
    preload_test:  bool,
    do_eval:       bool,
    save_best:     bool,
    resume_path:   str = None,
) -> None:
    if resume_path is not None:
        log.info(f"Resuming from {resume_path}")
        model = load_model_from_pretrained(model_name, resume_path, ArrhythmiaLabels.size)
    else:
        log.info(f"Creating new model")
        model = load_model(model_name)
        replace_classifier(model_name, model, ArrhythmiaLabels.size)    

    dataloader = build_dataloader(
        train_path    = train_path,
        test_path     = test_path,
        transform     = None,
        batch_size    = batch_size,
        preload_train = preload_train,
        preload_test  = preload_test,
    )

    save_name = f"{model_name}_ep{epochs}_i{image_size}"
    save_dir  = f"{save_dir_root}/{save_name}"

    log.info("Warming up dataloaders")
    warm_up_dataloader(dataloader["train"])
    warm_up_dataloader(dataloader["test"])

    training_stats = finetune(
        model      = model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth",
        do_eval    = do_eval,
        save_best  = save_best,
    )

    training_stats.serialize(f"{save_dir}/stats.txt")

    del dataloader


def train_model_distributed(
    rank:          int,
    world_size:    int,
    model_name:    str, 
    epochs:        int, 
    image_size:    int, 
    batch_size:    int, 
    train_path:    str,
    test_path:     str,
    save_dir_root: str,
    preload_train: bool,
    preload_test:  bool,
    do_eval:       bool,
    save_best:     bool,
    resume_path:   str = None,
) -> None:
    log.info(f"Running distributed training on rank {rank}")
    setup_distributed(rank, world_size)

    model = load_model(model_name)
    replace_classifier(model_name, model, ArrhythmiaLabels.size)

    model.to(rank)
    ddp_model = get_ddp_model(model, rank)

    dataloaders = build_dataloader(
        train_path    = train_path,
        test_path     = test_path,
        transform     = None,
        batch_size    = batch_size,
        preload_train = preload_train,
        preload_test  = preload_test,
        distributed   = True,
    )

    save_name = f"{model_name}_ep{epochs}_i{image_size}"
    save_dir  = f"{save_dir_root}/{save_name}"

    training_stats = finetune_distributed(
        ddp_model  = ddp_model,
        rank       = rank,
        epochs     = epochs,
        dataloader = dataloaders,
        save_path  = f"{save_dir}/{save_name}.pth",
        do_eval    = do_eval,
        save_best  = save_best,
    )

    training_stats.serialize(f"{save_dir}/stats_{rank}.txt")

    cleanup_distributed(rank)


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

    if config["distributed"]["do"]:
        if config["base_model"]["resume_path"] is not None:
            log.error("resume_path for distributed training is currently not supported")
            return

        run_distributed(
            train_model_distributed,
            world_size    = config["distributed"]["world_size"],
            model_name    = config["base_model"]["model_name"],
            epochs        = config["base_model"]["epochs"],
            image_size    = config["base_model"]["image_size"],
            batch_size    = config["base_model"]["batch_size"],
            train_path    = config["base_model"]["train_path"],
            test_path     = config["base_model"]["test_path"],
            save_dir_root = config["base_model"]["save_dir"],
            preload_train = config["base_model"]["preload_train"],
            preload_test  = config["base_model"]["preload_test"],
            do_eval       = config["base_model"]["evaluate"],
            save_best     = config["base_model"]["save_best"],
            resume_path   = config["base_model"]["resume_path"],
        )
    else:
        train_model(
            model_name    = config["base_model"]["model_name"],
            epochs        = config["base_model"]["epochs"],
            image_size    = config["base_model"]["image_size"],
            batch_size    = config["base_model"]["batch_size"],
            train_path    = config["base_model"]["train_path"],
            test_path     = config["base_model"]["test_path"],
            save_dir_root = config["base_model"]["save_dir"],
            preload_train = config["base_model"]["preload_train"],
            preload_test  = config["base_model"]["preload_test"],
            do_eval       = config["base_model"]["evaluate"],
            save_best     = config["base_model"]["save_best"],
            resume_path   = config["base_model"]["resume_path"],
        )

if __name__ == "__main__":
    main()
