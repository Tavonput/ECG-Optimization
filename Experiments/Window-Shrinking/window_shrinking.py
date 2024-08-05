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


def window_shrinking(
    model_name:    str, 
    save_dir_root: str,
    resume:        str = None
) -> None:
    """
    Progressive window shrinking. This will remain hard coded for now.
    """
    if resume is None:
        model = load_model(model_name)
        replace_classifier(model_name, model, ArrhythmiaLabels.size)
    else:
        model = load_model_from_pretrained(model_name, resume, ArrhythmiaLabels.size)

    TrainConfig  = namedtuple("TrainConfig", ["window_size", "ratio", "epochs", "batch_size"])
    train_config = [
        # TrainConfig(256, 0.3, 10, 128),
        TrainConfig(224, 0.3, 10, 128),
        TrainConfig(192, 0.4, 10, 128),
        TrainConfig(160, 0.5, 10, 128),
        TrainConfig(128, 1.0, 30, 128),
    ]

    for config in train_config:
        log.info(f"Window Size {config.window_size}")

        if config.window_size == 128:
            train_path = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_resample_i{config.window_size}_train.h5"
        else:
            train_path = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_resample_i{config.window_size}_train_r{config.ratio}.h5"

        dataloader = build_dataloader(
            train_path    = train_path,
            test_path     = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_resample_i128_test.h5",
            transform     = None,
            batch_size    = config.batch_size,
            preload_train = True,
            preload_test  = (config.window_size == 128),
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


def window_shrinking_multi_set(
    model_name:    str, 
    save_dir_root: str,
    resume:        str = None
) -> None:
    """
    Progressive window shrinking with full subset splits. This will remain hard coded for now.
    """
    if resume is None:
        model = load_model(model_name)
        replace_classifier(model_name, model, ArrhythmiaLabels.size)
    else:
        model = load_model_from_pretrained(model_name, resume, ArrhythmiaLabels.size)

    TrainConfig  = namedtuple("TrainConfig", ["window_size", "num_sets", "epochs", "batch_size"])
    train_config = [
        # TrainConfig(256, 1, 10, 128),
        TrainConfig(224, 1, 10, 128),
        TrainConfig(192, 1, 10, 128),
        TrainConfig(160, 1, 10, 128),
        TrainConfig(128, 1, 30, 128),
    ]

    for config in train_config:
        for s in range(config.num_sets):
            log.info(f"Window Size {config.window_size} - Set {s}")

            train_path = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_resample_i{config.window_size}_train.h5"
            # if config.window_size == 128:
            #     train_path = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_unfiltered_i{config.window_size}_train.h5"
            # else:
            #     train_path = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-{config.window_size}/image_unfiltered_i{config.window_size}_train.h5"

            dataloader = build_dataloader(
                train_path    = train_path,
                test_path     = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_resample_i128_test.h5",
                transform     = None,
                batch_size    = config.batch_size,
                preload_train = (config.window_size == 128),
                preload_test  = (config.window_size == 128)
            )

            save_name = f"{model_name}_ep{config.epochs}_i{config.window_size}"
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

#===========================================================================================================================
# Main
#
def main():
    window_shrinking(
        model_name    = "resnet18",
        save_dir_root = "../../Pretrained/ECG-Raw/Window-Shrinking/balance/Step-32/Resample",
        resume        = "../../Pretrained/ECG-Raw/Window-Shrinking/balance/Step-32/V1/resnet18_ep10_i256_r0.3/resnet18_ep10_i256_r0.3.pth",
    )

    # window_shrinking_multi_set(
    #     model_name    = "resnet18",
    #     save_dir_root = "../../Pretrained/ECG-Raw/Window-Shrinking/Full/Resample",
    #     resume        = "../../Pretrained/ECG-Raw/Window-Shrinking/Full/resnet18_ep10_i256/resnet18_ep10_i256.pth",
    # )


if __name__ == "__main__":
    main()