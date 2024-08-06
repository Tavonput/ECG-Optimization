import time
import os
import copy
import logging
import ast
from typing import List, Dict

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
    stats.total_train_time = training_time
    stats.serialize("stats.txt")
    stats.deserialize("stats.txt")
    ```
    """
    CURRENT_VERSION = 2

    def __init__(self, from_save: str = None) -> None:
        self.version = TrainingStats.CURRENT_VERSION

        self.best_accuracy    = 0.0
        self.total_train_time = 0.0

        self.running_accuracy   = []
        self.running_loss       = []
        self.running_train_time = []
        self.running_epoch_time = []
        
        self.epochs = 0

        if from_save is not None:
            self.deserialize(from_save)


    def serialize(self, path: str) -> None:
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        with open(path, "w") as file:
            file.write(f"{self.version}\n")
            file.write(f"{self.best_accuracy}\n")
            file.write(f"{self.running_accuracy}\n")
            file.write(f"{self.running_train_time}\n")
            file.write(f"{self.running_epoch_time}\n")
            file.write(f"{self.total_train_time}\n")
            file.write(f"{self.epochs}\n")
            file.write(f"{self.running_loss}")
    

    def deserialize(self, path: str) -> None:
        if not os.path.exists(path):
            log.error(f"{path} does not exist")
            return

        with open(path, "r") as file:
            lines = file.readlines()

            self.version = int(float(lines[0].strip()))
            if self.version != TrainingStats.CURRENT_VERSION:
                log.warning(f"{path} is not in up to date. Current: {self.version}, Newest: {TrainingStats.CURRENT_VERSION}")

            match self.version:
                case 1: 
                    self._deserialize_v1(path)
                case 2: 
                    self._deserialize_v2(path)
                case _:
                    log.error(f"Version {self.version} is not supported")
                

    def _deserialize_v1(self, path: str) -> None:
        with open(path, "r") as file:
            lines = file.readlines()

            self.best_accuracy      = float(lines[1].strip())
            self.running_accuracy   = ast.literal_eval(lines[2].strip())
            self.running_train_time = ast.literal_eval(lines[3].strip())
            self.running_epoch_time = ast.literal_eval(lines[4].strip())
            self.total_train_time   = float(lines[5].strip())
            self.epochs             = int(lines[6].strip())


    def _deserialize_v2(self, path: str) -> None:
        self._deserialize_v1(path)

        with open(path, "r") as file:
            lines = file.readlines()
            self.running_loss = ast.literal_eval(lines[7].strip())


    def display(self):
        log.info("Displaying TrainingStats")
        log.info(f"\tBest Accuracy:      {self.best_accuracy}")
        log.info(f"\tRunning Accuracy:   {self.running_accuracy}")
        log.info(f"\tRunning Train Time: {self.running_train_time}")
        log.info(f"\tRunning Epoch Time: {self.running_epoch_time}")
        log.info(f"\tTraining Time:      {self.total_train_time}")
        log.info(f"\tEpochs:             {self.epochs}")


class EvaluationStats:
    """
    EvaluationStats stores evaluation statistics and provides functions for serializing and deserializing the stats.

    Parameters
    ----------
    from_save : str
        A path to a serialized TrainingStats to load from.
    """
    CURRENT_VERSION = 1

    def __init__(self, from_save: str = None) -> None:
        self.version = EvaluationStats.CURRENT_VERSION

        self.accuracy:           float = 0.0
        self.per_class_accuracy: Dict[str, float] = {}
        self.precision:          Dict[str, float] = {}
        self.recall:             Dict[str, float] = {}
        self.f1_score:           Dict[str, float] = {}

        if from_save is not None:
            self.deserialize(from_save)


    def serialize(self, path: str) -> None:
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)

        with open(path, "w") as file:
            file.write(f"{self.version}\n")
            file.write(f"{self.accuracy}\n")
            file.write(f"{self.per_class_accuracy}\n")
            file.write(f"{self.precision}\n")
            file.write(f"{self.recall}\n")
            file.write(f"{self.f1_score}\n")


    def deserialize(self, path: str) -> None:
        if not os.path.exists(path):
            log.error(f"{path} does not exist")
            return
        
        with open(path, "r") as file:
            lines = file.readlines()

            self.version = int(lines[0].strip())
            if self.version != EvaluationStats.CURRENT_VERSION:
                log.warning(f"{path} is not in up to date")
                log.warning(f"\tNewest Version:  {EvaluationStats.CURRENT_VERSION}")
                log.warning(f"\tCurrent Version: {self.version}")

            match self.version:
                case 1: 
                    self._deserialize_v1(path)
                case _:
                    log.error(f"{self.version} is not supported")


    def _deserialize_v1(self, path: str) -> None:
        with open(path, "r") as file:
            lines = file.readlines()

            self.accuracy           = float(lines[1].strip())
            self.per_class_accuracy = ast.literal_eval(lines[2].strip())
            self.precision          = ast.literal_eval(lines[3].strip())
            self.recall             = ast.literal_eval(lines[4].strip())
            self.f1_score           = ast.literal_eval(lines[5].strip())


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
) -> List:
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
    
    Returns
    -------
    loss : List
        The running loss.
    """
    model.train()

    running_loss = []

    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        running_loss.append(loss.item())

        loss.backward()

        optimizer.step()
        scheduler.step()

    return running_loss


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

    per_class_accuracy       = (correct_samples_per_class / total_samples_per_class) * 100
    stats.per_class_accuracy = {f"class_{i}": per_class_accuracy[i].item() for i in range(num_classes)}
    stats.accuracy           = (total_correct / total_samples * 100).item()

    stats.precision = {f"class_{i}": precision(correct_samples_per_class[i].item(), false_positives[i].item()) for i in range(num_classes)}
    stats.recall    = {f"class_{i}": recall(correct_samples_per_class[i].item(), false_negatives[i].item()) for i in range(num_classes)}
    stats.f1_score  = {f"class_{i}": f1_score(stats.precision[f"class_{i}"], stats.recall[f"class_{i}"]) for i in range(num_classes)}

    return stats


def finetune(
    model:       nn.Module,
    epochs:      int,
    dataloader:  Dict[str, DataLoader],
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
    dataloader : Dict {str : DataLoader}
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
        running_loss     = train(model, dataloader["train"], criterion, optimizer, scheduler)
        train_end_time   = time.time()

        stats.running_train_time.append(train_end_time - train_start_time)
        stats.running_loss.append(running_loss)

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
                "epoch":                epoch + 1,
            }
            torch.save(checkpoint, f"{safety_dir}/check_{epoch+1}.pth")
            log.info(f"\tSaved epoch {epoch+1} to {safety_dir}/check_{epoch+1}.pth")
        
    end_time = time.time()
    log.info("Finetuning completed")

    stats.total_train_time = end_time - start_time

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


def precision(tp: float, fp: float) -> float:
    """
    Compute precision.
    """
    return tp / (tp + fp) if (tp + fp) > 0.0 else 0.0


def recall(tp: float, fn: float) -> float:
    """
    Compute recall.
    """
    return tp / (tp + fn) if (tp + fn) > 0.0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    """
    Compute f1 score.
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0