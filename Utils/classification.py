import time
import os
import copy
import logging
import ast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import *

from tqdm.auto import tqdm

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("CLASS").setLevel(logging.INFO)
log = logging.getLogger("CLASS")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#===========================================================================================================================
# Training And Eval
#
class TrainingStats:
    """
    TrainingStats stores training statistics and provides functions for serializing and deserializing the stats.

    Parameters
    ----------
    from_save : str
        A path to a serialized TrainingStats to load from.

    Example Usage
    -------------
    ```
    stats = TrainingStats()

    # During training...
    stats.best_accuracy = new_accuracy
    stats.running_accuracy.append(new_accuracy)

    # After training...
    stats.training_time = training_time
    stats.serialize("stats.txt")
    stats.deserialize("stats.txt")
    ```
    """
    def __init__(self, from_save: str = None) -> None:
        self.best_accuracy    = 0.0
        self.running_accuracy = []

        self.running_train_time = []
        self.running_epoch_time = []
        self.total_train_time   = 0.0

        self.epochs = 0

        if from_save is not None:
            self.deserialize(from_save)


    def serialize(self, path: str) -> None:
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        with open(path, "w") as file:
            file.write(f"{self.best_accuracy}\n")
            file.write(f"{self.running_accuracy}\n")
            file.write(f"{self.running_train_time}\n")
            file.write(f"{self.running_epoch_time}\n")
            file.write(f"{self.training_time}\n")
            file.write(f"{self.epochs}")
    

    def deserialize(self, path: str) -> None:
        if not os.path.exists(path):
            log.error(f"{path} does not exist")
            return

        with open(path, "r") as file:
            lines = file.readlines()
            self.best_accuracy      = float(lines[0].strip())
            self.running_accuracy   = ast.literal_eval(lines[1].strip())
            self.running_train_time = ast.literal_eval(lines[2].strip())
            self.running_epoch_time = ast.literal_eval(lines[3].strip())
            self.training_time      = float(lines[4].strip())
            self.epochs             = int(lines[5].strip())


    def display(self):
        log.info("Displaying TrainingStats")
        log.info(f"\tBest Accuracy:      {self.best_accuracy}")
        log.info(f"\tRunning Accuracy:   {self.running_accuracy}")
        log.info(f"\tRunning Train Time: {self.running_train_time}")
        log.info(f"\tRunning Epoch Time: {self.running_epoch_time}")
        log.info(f"\tTraining Time:      {self.training_time}")
        log.info(f"\tEpochs:             {self.epochs}")


class EvaluationStats:
    def __init__(self) -> None:
        self.accuracy:           float = 0.0
        self.per_class_accuracy: dict[str, float] = {}
        self.precision:          dict[str, float] = {}
        self.recall:             dict[str, float] = {}
        self.f1_score:           dict[str, float] = {}

    def display(self):
        log.info("Displaying EvaluationStats")
        log.info(f"\tAccuracy:  {self.accuracy}")
        
        for name in self.per_class_accuracy.keys():
            log.info(f"\tClass: {name}")
            log.info(f"\t\tAccuracy:  {self.per_class_accuracy[name]}")
            log.info(f"\t\tPrecision: {self.precision[name]}")
            log.info(f"\t\tRecall:    {self.recall[name]}")
            log.info(f"\t\tF1 Score:  {self.f1_score[name]}")


def train(
    model:      nn.Module,
    dataloader: DataLoader,
    criterion:  nn.Module,
    optimizer:  Optimizer,
    scheduler:  LambdaLR
) -> None:
    """
    Train a model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader : Dataloader
        Training dataloader.
    criterion : nn.Module
        Criterion.
    optimizer : Optimizer
        Optimizer.
    scheduler : LambdaLR
        Scheduler.
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

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : Dataloader
        Testing dataloader.
    verbose : bool
        Verbosity.

    Returns
    -------
    accuracy : float
        The accuracy
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


@torch.inference_mode()
def evaluate_per_class(
    model:       nn.Module,
    dataloader:  DataLoader,
    num_classes: int  = 5,
    verbose:     bool = True
) -> EvaluationStats:
    """
    Evaluate a model.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : Dataloader
        Testing dataloader.
    num_classes: int
        The number of classes.
    verbose : bool
        Verbosity.

    Returns
    -------
    stats : EvaluationStats
        The evaluation statistics.
    """
    model.eval()

    stats = EvaluationStats()
    total_samples_per_class   = torch.zeros(num_classes)
    correct_samples_per_class = torch.zeros(num_classes)
    total_samples             = 0
    total_correct             = 0

    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    for inputs, labels in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)

        for label in range(num_classes):
            total_samples_per_class[label]   += (labels == label).sum().item()
            correct_samples_per_class[label] += ((labels == label) & (outputs == label)).sum().item()
            false_positives[label]           += ((labels != label) & (outputs == label)).sum().item()
            false_negatives[label]           += ((labels == label) & (outputs != label)).sum().item()

        total_samples += labels.size(0)
        total_correct += (outputs == labels).sum()

    per_class_accuracy = (correct_samples_per_class / total_samples_per_class) * 100

    stats.per_class_accuracy = {f"class_{i}": per_class_accuracy[i].item() for i in range(num_classes)}
    stats.accuracy           = (total_correct / total_samples * 100).item()

    precision = correct_samples_per_class / (correct_samples_per_class + false_positives)
    recall    = correct_samples_per_class / (correct_samples_per_class + false_negatives)
    f1_score  = 2 * (precision * recall) / (precision + recall) 

    stats.precision = {f"class_{i}": precision[i].item() for i in range(num_classes)}
    stats.recall    = {f"class_{i}": recall[i].item() for i in range(num_classes)}
    stats.f1_score  = {f"class_{i}": f1_score[i].item() for i in range(num_classes)}

    return stats


def finetune(
    model:       nn.Module,
    epochs:      int,
    dataloader:  dict[str, DataLoader],
    save_path:   str,
    lr:          float = 0.01,
    safety:      int   = 0,
    safety_dir:  str   = None,
    do_eval:     bool  = True,
    save_best:   bool  = True,
    full_save:   bool  = False,
) -> TrainingStats:
    """
    Basic finetune implementation.

    Parameters
    ----------
    model : nn.Module
        Model to finetune.
    epochs : int
        Number of epochs to finetune.
    dataloader : dict {str : DataLoader}
        Both train and test loaders in dict {name, DataLoader}.
    save_path : str
        Where to save the model. 
    lr : float
        Initial learning rate for SGD.
    safety : int
        Epoch interval to save a checkpoint.
    safety_dir : str
        Save directory for safety checkpoints.
    do_eval : bool
        Whether or not to evaluate after every epoch.
    save_best : bool
        Whether or not to save the best epoch if do_eval is True.
    full_save : bool
        Save the entire model, not just the state dict.

    Returns
    -------
    stats : TrainingStats
        Statistics during training
    """
    if safety != 0 and not safety_dir:
        log.error("No safety directory specified")
        return
    
    if safety != 0 and not os.path.exists(safety_dir):
        os.makedirs(safety_dir)

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_model_checkpoint = dict()
    stats = TrainingStats()
    stats.epochs = epochs

    log.info("Begin finetuning...")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_start_time = time.time()
        train(model, dataloader["train"], criterion, optimizer, scheduler)
        train_end_time = time.time()

        stats.running_train_time.append(train_end_time - train_start_time)

        if do_eval:
            accuracy = evaluate(model, dataloader["test"])
            stats.running_accuracy.append(accuracy)

            if accuracy > stats.best_accuracy:
                if full_save:
                    best_model_checkpoint["model"] = copy.deepcopy(model)
                    
                best_model_checkpoint["state_dict"] = copy.deepcopy(model.state_dict())
                stats.best_accuracy = accuracy

            epoch_end_time = time.time()
            log.info(f"\tEpoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {stats.best_accuracy:.2f}%. Time: {(epoch_end_time - epoch_start_time)}s")
        else:
            epoch_end_time = time.time()
            log.info(f"\tEpoch {epoch+1} Finished. Time: {(epoch_end_time - epoch_start_time)}s")

        stats.running_epoch_time.append(epoch_end_time - epoch_start_time)

        # Checkpoint safety        
        if safety != 0 and (epoch + 1) % safety == 0:
            checkpoint = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch,
            }
            torch.save(checkpoint, f"{safety_dir}/check_{epoch+1}.pth")
            log.info(f"\tSaved epoch {epoch+1} to {safety_dir}/check_{epoch+1}.pth")
        
    end_time = time.time()
    log.info("Finetuning completed")

    stats.training_time = end_time - start_time

    if do_eval and save_best:
        if full_save:
            best_model_checkpoint["model"].zero_grad()
            torch.save(best_model_checkpoint["model"], save_path)
        else:
            torch.save(best_model_checkpoint["state_dict"], save_path)
    else:
        if full_save:
            model.zero_grad()
            torch.save(model, save_path)
        else:
            torch.save(model.state_dict(), save_path)

    log.info(f"Model saved to {save_path}")
    return stats


def warm_up_dataloader(dataloader, num_batches: int = 0) -> None:
    """
    Warmup a dataloader by iterating over the batches and doing nothing. Can help with caching.

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader to warmup.
    num_batches : int
        Number of batches to load. If not specified, all batches will be used.
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