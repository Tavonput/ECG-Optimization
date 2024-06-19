import random
import h5py
import os

import torch
import torch.nn as nn

from torchvision.transforms import *
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ArrhythmiaLabels:
    labels = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q",
    }
    size = 5

class EcgDataset(Dataset):
    def __init__(self, file_path: str, transform=None) -> None:
        super().__init__()

        self.h5_file   = h5py.File(file_path, "r")
        self.images    = self.h5_file["images"]
        self.labels    = self.h5_file["labels"]
        self.transform = transform

    def __del__(self):
        self.h5_file.close()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].astype(int)
        
        image = np.transpose(image, (1, 2, 0)) # (h, c, w) => (c, w, h)
        image = np.array(image, dtype=np.float32)

        label = np.array(label, dtype=np.int64)

        if self.transform:
            image = self.transform(image)

        return image, label
    
def build_dataloader(train_path: str, test_path: str, batch_size: int, transform) -> dict[str, DataLoader]:
    transforms = {
        "train": Compose([
            ToTensor(),
            transform,
        ]),
        "test": Compose([
            ToTensor(),
            transform,
        ])
    }

    dataset = {
        "train": EcgDataset(train_path, transform=transforms["train"]),
        "test":  EcgDataset(test_path, transform=transforms["test"]),       
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
