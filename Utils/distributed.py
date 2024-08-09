import os
import logging
import time
import copy
from typing import Dict, Callable

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import *
from torch.optim.lr_scheduler import *

from tqdm.auto import tqdm

from Dataset.dataset import DataLoaderSet
from .system import *
from .classification import TrainingStats, train


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("DISTR").setLevel(logging.INFO)
log = logging.getLogger("DISTR")


def init_process(rank: int, world_size: int, addr: str = "localhost", port: str = "12345") -> None:
    """
    Setup the distributed process group. Uses nccl as the backend.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.
    addr : str
        The address of the master process.
    port : str
        The communication port.
    """
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_process(rank: int = None):
    """
    Cleanup pytorch distributed process.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    """
    if rank is not None:
        log.info(f"Cleaning up processes group from rank {rank}")

    dist.destroy_process_group()


def _run_distributed_wrapper(rank, world_size, func, kwargs):
    """
    Wrapper function for the run_distributed function.
    """
    func(rank, world_size, **kwargs)


def run_distributed(func: Callable, world_size: int, **kwargs) -> None:
    """
    Run a function in a distributed manner.

    Parameters
    ----------
    func : Callable
        The function to run in each process.
    world_size : int
        The number of processes.
    **kwargs
        Additional arguments passes to the function.
    """
    mp.spawn(
        _run_distributed_wrapper,
        args   = (world_size, func, kwargs,),
        nprocs = world_size,
        join   = True,
    )


def get_ddp_model(model: nn.Module, rank: int) -> nn.Module:
    """
    Get the DDP model. Wrapper around torch.nn.parallel.DistributedDataParallel.

    Parameters
    ----------
    model : nn.Module
        The model to use.
    rank : int
        The rank of the current process.
    """
    return DDP(model, device_ids=[rank])

@torch.inference_mode()
def evaluate_distributed(
    model:      nn.Module,
    dataloader: DataLoader,
    rank:       str,
    verbose:    bool = True
) -> float:
    """
    Evaluate a model in a distributed manner.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : Dataloader
        Testing dataloader.
    rank : str
        The rank of the current process.
    verbose : bool
        Verbosity.

    Returns
    -------
    accuracy : float
        The accuracy
    """
    model.eval()

    num_samples = torch.tensor(0, dtype=torch.float32, device=rank, requires_grad=False)
    num_correct = torch.tensor(0, dtype=torch.float32, device=rank, requires_grad=False)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Eval", leave=False, disable=not verbose):
            inputs = inputs.to(rank)
            labels = labels.to(rank)

            outputs = model(inputs)
            outputs = outputs.argmax(dim=1)

            num_samples += labels.size(0)
            num_correct += (outputs == labels).sum()

    dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)

    return num_correct.item() / num_samples.item() * 100


def finetune_distributed(
    ddp_model:   nn.Module,
    rank:        int,
    epochs:      int,
    dataloader:  DataLoaderSet,
    save_path:   str,
    lr:          float = 0.01,
    do_eval:     bool  = True,
    save_best:   bool  = True,
    full_save:   bool  = False,
) -> TrainingStats:
    """
    Distributed finetune implementation. Based off Utils.classification.finetune.

    Parameters
    ----------
    ddp_model : nn.Module
        The DDP model to finetune.
    rank : int
        The rank of the current process.
    epochs : int
        Number of epochs to finetune.
    dataloader : DataLoaderSet
        The dataloaders and the training sampler.
    save_path : str
        Where to save the model. 
    lr : float
        Initial learning rate for SGD.
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
    if rank == 0:
        check_path_for_dir(save_path, create=True)

    optimizer = SGD(ddp_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_model_checkpoint = dict()
    stats = TrainingStats()
    stats.epochs = epochs

    dist.barrier()

    log.info(f"Rank {rank}: Begin finetuning...")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        dataloader.train_sampler.set_epoch(epoch)

        train_start_time = time.time()
        running_loss     = train(ddp_model, dataloader.train_loader, criterion, optimizer, scheduler, rank)
        train_end_time   = time.time()

        stats.running_train_time.append(train_end_time - train_start_time)
        stats.running_loss.append(running_loss)

        if do_eval:
            accuracy = evaluate_distributed(ddp_model.module, dataloader.test_loader, rank)
            stats.running_accuracy.append(accuracy)

            if accuracy > stats.best_accuracy:
                if rank == 0:
                    if full_save:
                        best_model_checkpoint["model"] = copy.deepcopy(ddp_model)
                        
                    best_model_checkpoint["state_dict"] = copy.deepcopy(ddp_model.state_dict())
                stats.best_accuracy = accuracy

            dist.barrier()
            epoch_end_time = time.time()
            log.info(f"\tRank {rank}: Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {stats.best_accuracy:.2f}%. Time: {(epoch_end_time - epoch_start_time)}s")
        else:
            epoch_end_time = time.time()
            log.info(f"\tRank {rank}: Epoch {epoch+1} Finished. Time: {(epoch_end_time - epoch_start_time)}s")

        stats.running_epoch_time.append(epoch_end_time - epoch_start_time)

    end_time = time.time()
    log.info(f"Rank {rank}: Finetuning completed")

    stats.total_train_time = end_time - start_time

    if rank == 0:
        if do_eval and save_best:
            if full_save:
                best_model_checkpoint["model"].zero_grad()
                torch.save(best_model_checkpoint["model"], save_path)
            else:
                torch.save(best_model_checkpoint["state_dict"], save_path)
        else:
            if full_save:
                ddp_model.zero_grad()
                torch.save(ddp_model, save_path)
            else:
                torch.save(ddp_model.state_dict(), save_path)

        log.info(f"Rank {rank}: Model saved to {save_path}")

    return stats
