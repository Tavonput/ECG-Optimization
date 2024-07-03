import random
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances

from pyts.image import GramianAngularField


#===========================================================================================================================
# Data Generation from CSV (Kinda of deprecated. Use data_generation.py instead)
#
# https://www.kaggle.com/datasets/shayanfazeli/heartbeat
#
class ArrhythmiaLabels:
    labels = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q",
    }
    size = 5


def visualize_raw_ecg(data: dict[str, pd.DataFrame]) -> None:
    """
    Summary.

    Parameters
    ----------
    """
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    features = data["features"]
    labels   = data["labels"]

    example_idx = []
    for label in range(ArrhythmiaLabels.size):
        indices = labels[labels == label].index
        index   = random.choice(indices)
        example_idx.append(index)

    example_features = features.iloc[example_idx]

    _, ax = plt.subplots(1, ArrhythmiaLabels.size, figsize=(20, 4))
    for i in range(ArrhythmiaLabels.size):
        ax[i].plot(example_features.iloc[i])
        ax[i].set_title(f"{ArrhythmiaLabels.labels[i]}")
        ax[i].grid(False)

    plt.show()


def visualize_image(image):
    """
    Visualize an image in grayscale.

    Parameters
    ----------
    image : 
        The image.
    """
    plt.imshow(image, cmap="gray")
    plt.show()


def gaf(samples):
    """
    Summary.

    Parameters
    ----------
    """
    gaf = GramianAngularField(method="summation")
    return gaf.fit_transform(samples)


def recurrence_plot(samples, eps: float = 0.1, steps: int = 10):
    """
    Summary.

    Parameters
    ----------
    """
    results = []
    for s in samples.values:
        d = pairwise_distances(s[:, None])
        d = d / eps
        d[d > steps] = steps
        results.append(d/5. - 1)
    return results


def get_quantiles(min_value=0, max_val=1, k=10):
    """
    Summary.

    Parameters
    ----------
    """
    c = (max_val - min_value)/k
    b = min_value + c
    d = []
    for i in range(1, k):
        d.append(b)
        b += c
    d.append(max_val)
    return d


def value_to_quantile(x):
    """
    Summary.

    Parameters
    ----------
    """
    quantiles = get_quantiles()
    for i, k in enumerate(quantiles):
        if x <= k:
            return i
    return 0

def mtf(samples, size=10):
    """
    Summary.

    Parameters
    ----------
    """
    results = []

    for x in samples.values:
        q = np.vectorize(value_to_quantile)(x)
        r = np.zeros((q.shape[0], q.shape[0]))
        y = np.zeros((size, size))
        for i in range(x.shape[0] - 1):
            y[q[i], q[i + 1]] += 1
        y = y / y.sum(axis=1, keepdims=True)
        y[np.isnan(y)] = 0
        
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r[i, j] = y[q[i], q[j]]
        
        results.append(r/5. - 1)
        
    return results


def batch_value(value: int, batch_size: int) -> list[int]:
    """
    Calculate batching. For example if value is 33 and the batch size is 10, the corresponding batching will be [10, 10, 10, 3].

    Parameters
    ----------
    value : int
        Number of samples to batch.
    batch_size : int
        The batch_size.

    Returns
    -------
    batches : list[int]
        Batching list.
    """
    full_batches = [batch_size] * (value // batch_size)
    remainder    = value % batch_size

    if remainder != 0:
        full_batches += [remainder]
        
    return full_batches



def transform_data(output_path: str, data: dict[str, pd.DataFrame], batch_size: int) -> None:
    """
    Summary.

    Parameters
    ----------
    """
    features_df = data["features"]
    labels_df   = data["labels"]
    num_samples = features_df.shape[0]
    image_size  = features_df.shape[1]

    print(f"Transforming {num_samples} samples...")

    # Initialize dataset for images and labels
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset(
            "images", 
            shape=(0, 3, image_size, image_size), 
            maxshape=(None, 3, image_size, image_size), 
            chunks=(1, 3, image_size, image_size), 
            dtype=np.float32
        )
        
        hf.create_dataset(
            "labels", 
            shape=(0,), 
            maxshape=(None,), 
            chunks=(1,), 
            dtype=np.float32
        )
        
    # Transform samples in batches
    batches = batch_value(num_samples, batch_size)
    current_idx = 0
    for batch in batches:
        
        # Transform the samples
        samples     = features_df[current_idx : (current_idx + batch)]
        samples_gaf = gaf(samples)
        samples_rp  = recurrence_plot(samples)
        samples_mtf = mtf(samples)

        # Stack into 3 channels
        images_trans = np.stack((samples_gaf, samples_rp, samples_mtf), axis=0) # (3, batch_size, width, height)
        images_trans = images_trans.transpose(1, 0, 2, 3) # (batch_size, 3, width, height)

        labels_trans = labels_df[current_idx : (current_idx + batch)]

        # Store the images
        with h5py.File(output_path, "a") as hf:
            images = hf["images"]
            labels = hf["labels"]

            images.resize(images.shape[0] + batch, axis=0)
            labels.resize(labels.shape[0] + batch, axis=0)

            for i, (image, label) in enumerate(zip(images_trans, labels_trans)):
                images[i + current_idx] = image
                labels[i + current_idx] = label

        print(f"    Completed images {current_idx}-{current_idx + batch}")

        current_idx += batch
