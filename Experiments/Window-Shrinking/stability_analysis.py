import sys
sys.path.append("../../")

import logging
from collections import namedtuple

import torch

from Dataset.data_generation import ArrhythmiaLabels
from Dataset.dataset import build_dataloader

from Utils.model_loading import *
from Utils.classification import warm_up_dataloader, finetune


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("WINSH").setLevel(logging.INFO)
log = logging.getLogger("WINSH")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_per_epoch_save(
    model_name:    str, 
    save_dir_root: str,
    image_size:    int,
) -> None:
    """
    Saves on each epoch. This will remain hard coded for now.
    """
    model = load_model(model_name)
    replace_classifier(model_name, model, ArrhythmiaLabels.size)

    epochs = 10

    dataloader = build_dataloader(
        train_path    = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{image_size}/image_unfiltered_i{image_size}_train_r0.3_balance.h5",
        test_path     = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{image_size}/image_unfiltered_i{image_size}_test.h5",
        transform     = None,
        batch_size    = 128,
        preload_train = True,
    )

    save_name = f"{model_name}_ep{epochs}_i{image_size}"
    save_dir  = f"{save_dir_root}/{save_name}"

    warm_up_dataloader(dataloader["train"])
    warm_up_dataloader(dataloader["test"])

    training_stats = finetune(
        model      = model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth",
        do_eval    = True,
        safety     = 1,
        safety_dir = save_dir
    )

    training_stats.serialize(f"{save_dir}/stats.txt")

    del dataloader


def train_from_save(
    model_name:    str,
    save_dir_root: str,
    save_name:     str, 
    resume:        str,
    image_size:    int,
) -> None:
    """
    Train a model from a save. This will remain hard coded for now.
    """
    model = load_model_from_pretrained(model_name, resume, ArrhythmiaLabels.size)

    epochs = 20

    dataloader = build_dataloader(
        train_path    = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{image_size}/image_unfiltered_i{image_size}_train.h5",
        test_path     = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{image_size}/image_unfiltered_i{image_size}_test.h5",
        transform     = None,
        batch_size    = 128,
        preload_train = True,
        preload_test  = True,
    )

    save_dir  = f"{save_dir_root}/{save_name}"

    warm_up_dataloader(dataloader["train"])
    warm_up_dataloader(dataloader["test"])

    training_stats = finetune(
        model      = model,
        epochs     = epochs,
        dataloader = dataloader,
        save_path  = f"{save_dir}/{save_name}.pth",
        do_eval    = True,
    )

    training_stats.serialize(f"{save_dir}/stats.txt")

    del dataloader


#===========================================================================================================================
# Main
#
def main():
    # train_per_epoch_save(
    #     model_name    = "resnet18",
    #     save_dir_root = "../../Pretrained/ECG-Raw/Window-Shrinking/Stability-Analysis",
    #     image_size    = 256
    # )

    for i in range(10):
        train_from_save(
            model_name    = "resnet18",
            save_dir_root = "../../Pretrained/ECG-Raw/Window-Shrinking/Stability-Analysis",
            save_name     = f"resnet18_ep20_i128_{i+1}",
            resume        = f"../../Pretrained/ECG-Raw/Window-Shrinking/Stability-Analysis/resnet18_ep10_i256/check_{i+1}.pth",
            image_size    = 128,
        )


if __name__ == "__main__":
    main()