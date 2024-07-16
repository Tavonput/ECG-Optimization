import time
import random
import h5py
import logging

import torch

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt

import psutil


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("DTSET").setLevel(logging.INFO)
log = logging.getLogger("DTSET")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EcgDataset(Dataset):
    """
    @class ECG Dataset

    ECG dataset from MIT-BIH transformed into images and loaded from a HDF5 file. To be used with PyTorch DataLoader.

    Parameters
    ----------
    file_path : str
        Path to HDF5 dataset file.
    transform : A transform from PyTorch
        Additional tensor transforms to apply.
    preload : bool
        Whether or not the data should be preloaded host memory.
    half_precision : bool 
        Whether or not the data should sent to FP16
    """
    def __init__(self, file_path: str, transform = None, preload: bool = False, half_precision: bool = False, max_memory: float = 0.95) -> None:
        super().__init__()
        log.info(f"Building ECG Dataset from {file_path}")

        self.h5_file   = h5py.File(file_path, "r")
        self.images    = self.h5_file["images"]
        self.labels    = self.h5_file["labels"]
        
        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        self.transform      = transform
        self.preload        = preload
        self.half_precision = half_precision

        self.max_memory = max_memory
        memory_okay     = self._check_memory_usage()

        if preload:
            if memory_okay is False:
                log.warning("\tUsing too much memory. Disabling preload")
            else:
                log.info("\tPreloading images and labels")
                self.preloaded_images = [i for i in self.images]
                self.preloaded_labels = [l for l in self.labels]


    def __del__(self):
        self.h5_file.close()


    def __len__(self):
        return self.num_images
    

    def __getitem__(self, idx):
        if self.preload:
            image = self.preloaded_images[idx]
            label = self.preloaded_labels[idx].astype(int)
        else:
            image = self.images[idx]
            label = self.labels[idx].astype(int)
        
        image = np.transpose(image, (1, 2, 0)) # (h, c, w) => (c, w, h)
        
        if self.half_precision:
            image = np.array(image, dtype=np.float16)
        else:
            image = np.array(image, dtype=np.float32)

        label = np.array(label, dtype=np.int64)

        if self.transform:
            image = self.transform(image)

        return image, label
    

    def _check_memory_usage(self) -> bool:
        """
        Check if there is enough memory available.
        """
        memory_info = psutil.virtual_memory()
        available   = memory_info.available
        needed      = self._compute_memory_usage()

        log.info(f"\tAvailable Memory: {(available / 1024 ** 3):.2f} GiB")
        log.info(f"\tDataset Memory:   {(needed / 1024 ** 3):.2f} GiB")
        
        left_over = available - needed
        threshold = memory_info.total * (1 - self.max_memory)
        if left_over <= threshold:
            log.warning(f"\tUsing too much memory (Left Over: {(left_over / 1024 ** 3):.2f}, Safety: {(threshold / 1024 ** 3):.2f}). Preloading or caching might struggle")
            return False
        
        return True


    def _compute_memory_usage(self) -> float:
        """
        Compute the total number of bytes for the dataset.
        """
        images_dtype = self.images.dtype
        labels_dtype = self.labels.dtype

        num_image_elements = np.prod(self.images.shape, dtype=np.float64)
        num_label_elements = np.prod(self.labels.shape, dtype=np.float64)

        images_memory = num_image_elements * images_dtype.itemsize
        labels_memory = num_label_elements * labels_dtype.itemsize

        return images_memory + labels_memory
    

def build_dataloader(
    train_path:     str, 
    test_path:      str, 
    batch_size:     int, 
    transform, 
    preload_train:  bool = False,
    preload_test:   bool = False,
    half_precision: bool = False,
    max_memory:     float = 0.95
) -> dict[str, DataLoader]:
    """
    Build the train and test dataloaders from EcgDataset.

    Parameters
    ----------
    train_path : str
        Path to HDF5 training set.
    test_path : str
        Path to HDF5 test set.
    batch_size : int
        Batch size for train and test sets.
    transform : A Pytorch transform
        PyTorch transform to apply on the images after tensor conversion.
    preload_train : bool
        Whether or not the training data should be preloaded into host memory.
    preload_test : bool
        Whether or not the testing data should be preloaded into host memory.
    half_precision : bool
        Whether or not the data should be loaded as FP16.
    max_memory : float
        Ratio of for the maximum amount of memory before sending a warning.
        
    Returns
    -------
    dataloaders : dict {str : DataLoader}
        Dictionary containing two entries:
            "train": Training dataloader
            "test":  Testing dataloader

    TODO: Transforms provided by the user should be a list of transforms rather than just one.
    """
    if transform is None:
        trans = Compose([ToTensor()])
    elif isinstance(transform, list):
        trans = Compose([ToTensor()] + transform)
    else:
        trans = Compose([ToTensor(), transform])

    transforms = {
        "train": trans,
        "test":  trans
    }

    dataset = {
        "train": EcgDataset(train_path, transform=transforms["train"], preload=preload_train, half_precision=half_precision, max_memory=max_memory),
        "test":  EcgDataset(test_path, transform=transforms["test"], preload=preload_test, half_precision=half_precision, max_memory=max_memory),       
    }

    dataloader = {}
    for split in ["train", "test"]:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = 0,
            pin_memory  = True,
        )

    return dataloader


def visualize_ecg_data(dataloader: DataLoader) -> None:
    """
    Visualize the transformed ECG data.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader for the ECG data.
    """
    batch = next(iter(dataloader))
    num_images = len(batch[0])

    idx = random.choice(range(0, num_images))
    image = batch[0][idx]
    channels = torch.split(image, 1, dim=0)

    _, ax = plt.subplots(1, 3)
    for i, channel in enumerate(channels):
        ax[i].imshow(channel.squeeze(0).numpy(), cmap="gray")
        ax[i].axis("off")
    plt.show()


def examine_dataset(dataset_path: str) -> dict[str, int]:
    """
    Examine the keys in a hdf5 file.

    Parameters
    ----------
    dataset_path : str 
        Path to hdf5 file.
    
    Returns
    -------
    stats : dict {str : int}
        Dataset information in the form of a dictionary.
    """
    data = dict()
    with h5py.File(dataset_path, "r") as hdf:
        for key in hdf.keys():
            data[f"{key}_size"] = len(hdf[key])
            data[f"{key}_shape"] = hdf[key].shape
            data[f"{key}_dtype"] = hdf[key].dtype
            
    return data


def benchmark_dataloader(loader: DataLoader, num_batches: int = 100) -> float:
    # Warmup for cache
    for i, batch in enumerate(loader):
        if i >= num_batches - 1:
            break

    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches - 1:
            break

    end_time = time.time()
    return (end_time - start_time) / num_batches

#===========================================================================================================================
# Main (Used for testing this file)
#
if __name__ == "__main__":
    import sys
    sys.path.append("../")

    dataloader = build_dataloader(
        train_path = "../Data/MIT-BIH-Raw/Datasets/Resolution-64/image_unfiltered_i64_train.h5",
        test_path  = "../Data/MIT-BIH-Raw/Datasets/Resolution-64/image_unfiltered_i64_test.h5",
        transform  = None,
        batch_size = 128,
        preload    = True
    )

    print("Benchmarking...")
    print(benchmark_dataloader(dataloader["test"], 100))