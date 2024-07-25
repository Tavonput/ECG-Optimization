import sys
sys.path.append("../../")

import logging

import torch

from Dataset.data_generation import ArrhythmiaLabels
from Dataset.dataset import build_dataloader

from Utils.model_loading import *
from Utils.classification import warm_up_dataloader, finetune
from Utils.benchmarking import evaluate, add_model_stat_to_json, ModelStats


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("DWNSP").setLevel(logging.INFO)
log = logging.getLogger("DWNSP")

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
    preload_test:  bool,
    do_eval:       bool,
    save_best:     bool,
    resume_path:   str = None,
) -> None:
    """
    Base model training.
    """
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


def average_model_weights(models: list[nn.Module], new_model: nn.Module) -> None:
    """
    Average the weights of a list of models and store them in a new model.
    """
    for model in models:
        model.eval()
        model.to("cpu")

    new_state_dict = new_model.state_dict()

    for key in new_state_dict:
        new_state_dict[key] = torch.zeros_like(new_state_dict[key])

    with torch.no_grad():
        for model in models:
            state_dict = model.state_dict()
            for key in state_dict:
                new_state_dict[key] += state_dict[key]

        for key in new_state_dict:
            if torch.is_floating_point(new_state_dict[key]):
                new_state_dict[key] /= len(models)
            else:
                new_state_dict[key] //= len(models)

    new_model.load_state_dict(new_state_dict)


def train_multiple_models():
    """
    Train multiple models on different training sets. This will remain hard coded for now.
    """
    num_sets = [2, 3, 4, 5]

    for s in num_sets:
        for i in range(s):
            base_model(
                model_name    = "resnet18",
                epochs        = 20,
                image_size    = 128,
                batch_size    = 128,
                train_path    = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-128/unfiltered_i128_train_full_split_{s}/image_set_{i}.h5",
                test_path     = "../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_unfiltered_i128_test.h5",
                save_dir_root = f"../../Pretrained/ECG-Raw/Base/Full-Split/Multi/Sets-{s}/Set-{i}",
                preload_train = True,
                preload_test  = True,
                do_eval       = True,
                save_best     = False,
            )


def train_single_model():
    """
    Train a single model on a sequence of training sets. This will remain hard coded for now.
    """
    num_sets = [2, 3, 4, 5]

    for s in num_sets:
        for i in range(s):
            if i == 0:
                resume_path = None
            else:
                resume_path = f"../../Pretrained/ECG-Raw/Base/Full-Split/Single/Sets-{s}/Set-{i - 1}/resnet18_ep20_i128/resnet18_ep20_i128.pth"                

            base_model(
                model_name    = "resnet18",
                epochs        = 20,
                image_size    = 128,
                batch_size    = 128,
                train_path    = f"../../Data/MIT-BIH-Raw/Datasets/Resolution-128/unfiltered_i128_train_full_split_{s}/image_set_{i}.h5",
                test_path     = "../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_unfiltered_i128_test.h5",
                save_dir_root = f"../../Pretrained/ECG-Raw/Base/Full-Split/Single/Sets-{s}/Set-{i}",
                preload_train = True,
                preload_test  = True,
                do_eval       = True,
                save_best     = (i == s - 1), # Last iteration
                resume_path   = resume_path,
            )


def combine_model_weights():
    """
    Combine the weights from 'train_multiple_models' into a new model. This will remain hard coded for now.
    """
    num_sets = [2, 3, 4, 5]

    for s in num_sets:
        base_dir = f"../../Pretrained/ECG-Raw/Base/Full-Split/Multi/Sets-{s}"

        models = []
        for i in range(s):
            model_path = f"{base_dir}/Set-{i}/resnet18_ep20_i128/resnet18_ep20_i128.pth"
            model      = load_model_from_pretrained("resnet18", model_path, ArrhythmiaLabels.size)
            models.append(model)

        new_model = resnet18()
        replace_classifier("resnet18", new_model, ArrhythmiaLabels.size)
        average_model_weights(models, new_model)

        dataloader = build_dataloader(
            train_path    = "../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_unfiltered_i128_train.h5",
            test_path     = "../../Data/MIT-BIH-Raw/Datasets/Resolution-128/image_unfiltered_i128_test.h5",
            transform     = None,
            batch_size    = 128,
        )

        new_model.to(device)
        accuracy = evaluate(new_model, dataloader["test"])
        stats    = ModelStats(name=f"ResNet18 - Combined {s}", accuracy=accuracy)

        torch.save(new_model.state_dict(), f"{base_dir}/combined.pth")
        add_model_stat_to_json(f"{base_dir}/combined_stats.json", stats)


#===========================================================================================================================
# Main
#
def main():
    # train_single_model()
    # train_multiple_models()
    combine_model_weights()

    return


if __name__ == "__main__":
    main()
