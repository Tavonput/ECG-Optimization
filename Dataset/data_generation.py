import sys
sys.path.append("../")

import h5py
import wfdb
import logging
import os
import time
from typing import Union, Tuple, List, Dict

import numpy as np

from scipy.signal import resample
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
import cv2

from Utils.system import *


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("DTGEN").setLevel(logging.INFO)
log = logging.getLogger("DTGEN")


#===========================================================================================================================
# Useful Stuff for Navigating the Dataset
#
class ArrhythmiaLabels:
    """
    Constants and label conversions related to the arrhythmia database.
    """
    size = 5

    classes = ["normal", "sveb", "veb", "fusion", "unknown"]
    records = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113',
               '114', '115', '116', '117', '118', '119', '121', '122', '123', '124',
               '200', '201', '202', '203', '205', '207', '208', '209', '210', '212',
               '213', '214', '215', '219', '220', '221', '222', '223', '228', '230',
               '231', '232', '233', '234']
    
    records_full = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', 
                    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', 
                    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', 
                    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221', 
                    '222', '223', '228', '230', '231', '232', '233', '234']

    raw_to_idx = {
        'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, # Normal
        'A': 1, 'a': 1, 'J': 1, 'S': 1,         # SVEB
        'V': 2, 'E': 2,                         # VEB
        'F': 3,                                 # Fusion
        'Q': 4, '/': 4, 'f': 4                  # Unknown
    }

    idx_to_class = {
        0: "normal",
        1: "sveb",
        2: "veb",
        3: "fusion",
        4: "unknown"
    }

    class_to_idx = {
        "normal":  0,
        "sveb":    1,
        "veb":     2,
        "fusion":  3,
        "unknown": 4
    }


#===========================================================================================================================
# Data Generation Helpers
#
def segment_heartbeat(signal: np.ndarray, peak: int, window_size: int) -> Union[np.ndarray, None]:
    """
    Segment a heartbeat. The length of the segment will be 2 x window_size.

    Parameters
    ----------
    signal : np.ndarray
        The ECG signal.
    peak : int
        The r-peak of the heartbeat.
    window_size : int
        The window size.

    Returns
    -------
    segment : np.ndarray | None
        The heartbeat segment or None if the segment is outside of the ECG signal.
    """
    start_idx = peak - window_size
    end_idx   = peak + window_size
    
    if start_idx >= 0 and end_idx < len(signal):
        return signal[start_idx : end_idx]
    
    return None


def process_record(record_path: str, window_size: int) -> Tuple[List, List]:
    """
    Process a record given a window size.

    Parameters
    ----------
    record_path : str
        Path to the record.
    window_size : int
        The window size (total size).

    Returns
    -------
    segments : List, List
        Array of heartbeat segments and an array of labels.
    """
    record     = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    ecg_signal = record.p_signal[:, 0]

    r_peaks = annotation.sample
    labels  = annotation.symbol

    window_size_half = window_size // 2
    heartbeats = []
    labels_idx = []
    for peak, label in zip(r_peaks, labels):
        if label in ArrhythmiaLabels.raw_to_idx:
            segment = segment_heartbeat(ecg_signal, peak, window_size_half)

            if segment is not None:
                heartbeats.append(segment)
                labels_idx.append(ArrhythmiaLabels.raw_to_idx[label])
    
    return np.array(heartbeats), np.array(labels_idx)


def get_class_distribution(labels: List, normalize: bool = False) -> Dict[str, Union[int, float]]:
    """
    Get the class distribution of a list of labels.

    Parameters
    ----------
    labels : List
        List of labels.
    normalize : bool
        Whether or not to normalize the distribution.

    Returns
    -------
    distribution : Dict {str ; int|float}
        Class distribution in the form of a dictionary (class name ; count/ratio).
    """
    num_labels = len(labels)

    counts = dict()
    for i, label in enumerate(ArrhythmiaLabels.classes):
        counts[label] = np.sum(labels == i)

        if normalize:
            counts[label] /= num_labels
        
    return counts


def batch_value(value: int, batch_size: int) -> List[int]:
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
    batches : List[int]
        Batching list.
    """
    full_batches = [batch_size] * (value // batch_size)
    remainder    = value % batch_size

    if remainder != 0:
        full_batches += [remainder]
        
    return full_batches


def signal_to_image(signals: np.ndarray) -> np.ndarray:
    """
    Transform signals into images. Signals should be of shape (num_samples, image_size).

    Parameters
    ----------
    signals : np.ndarray
        The 1D signals to transform.

    Returns
    -------
    images : np.ndarray
        The transformed images of shape (num_samples, 3, image_size, image_size).
    """
    gaf = GramianAngularField(method="summation")
    gaf_image = gaf.fit_transform(signals)

    rp = RecurrencePlot(threshold="point")
    rp_image = rp.fit_transform(signals)
    
    mtf = MarkovTransitionField()
    mtf_image = mtf.fit_transform(signals)

    images = np.stack((gaf_image, rp_image, mtf_image), axis=0) # (3, batch_size, width, height)
    return images.transpose(1, 0, 2, 3) # (batch_size, 3, width, height)


def split_value(value: int, ratio: float, shuffle: bool) -> Tuple[List[int], List[int]]:
    """
    Split a value by a ratio into indices. For example, a value of 5 with a ratio of 0.6 will results in two lists:
    [0, 1, 2] and [3, 4].

    Parameters
    ----------
    value : int
        The value to split into indices.
    ratio : float
        The split ratio.
    shuffle : bool
        Whether or not to shuffle the indices before the split.

    Returns
    -------
    indices : List[int], List[int]
        Two indices lists.
    """
    indices = np.arange(value)

    if shuffle:
        np.random.shuffle(indices)

    split_idx   = int(value * ratio)
    train_split = indices[:split_idx]
    test_split  = indices[split_idx:]

    return train_split, test_split


def remove_normals(labels: np.ndarray, ratio_to_keep: float, min_to_keep: int) -> Union[List[int], None]:
    """
    Given an array of labels, create a subset by only removing instances of the normal label.

    Parameters
    ----------
    labels : np.ndarray
        The labels.
    ratio_to_keep : float
        The ratio to make the subset.
    min_to_keep : int
        The minimum number of normals to keep.
    
    Returns
    -------
    indices : List[int] | None
        The indices of labels to keep or None if too many normals were attempted to remove.
    """
    num_labels         = len(labels)
    num_labels_to_keep = int(num_labels * ratio_to_keep)

    normal_indices = np.where(labels == 0)[0]
    num_normals    = len(normal_indices)
    num_abnormal   = num_labels - num_normals
    np.random.shuffle(normal_indices)

    num_normals_to_remove = num_normals - (num_labels_to_keep - num_abnormal)

    if num_normals_to_remove >= num_normals - min_to_keep:
        log.error(f"Too many normals to remove. Left: {num_normals - num_normals_to_remove}, Minimum: {min_to_keep}")
        return None
    
    # Remove the normal labels
    indices_to_keep = np.setdiff1d(np.arange(num_labels), normal_indices[:num_normals_to_remove])
    return indices_to_keep


def random_subset(value: int, ratio: float, shuffle: bool) -> List[int]:
    """
    Randomly select some ratio of indices from a value. If the value is high enough, the random sampling should be maintain
    the original distribution.

    Parameters
    ----------
    value : int
        The value to select from.
    ratio : float
        The ratio to make the subset.
    shuffle : bool
        Whether or not to shuffle the indices before selecting.

    Returns
    -------
    indices : List[int]
        The indices.
    """
    indices = np.arange(value)

    if shuffle:
        np.random.shuffle(indices)

    split_inx = int(value * ratio)
    return indices[:split_inx]


class SamplePreprocessor:
    """
    Namespace for sample preprocessing.
    """
    def preprocess(data: np.ndarray, method: str, data_type: str, new_size: int) -> Union[np.ndarray, None]:
        """
        Preprocess either images or a signals.

        Parameters
        ----------
        data : np.ndarray
            The data to be preprocessed.
        method : str
            The preprocessing method.
        data_type : str
            The data type.
        new_size : int
            The size of the preprocessed data.

        Returns
        -------
        new_data : np.ndarray
            The preprocessed data.
        None
            If the preprocessing failed.
        """
        if data_type == "image":
            return SamplePreprocessor._preprocess_image(data, method, new_size)
        elif data_type == "signal":
            return SamplePreprocessor._preprocess_signal(data, method, new_size)
        else:
            log.error(f"{data_type} is not a valid sample type for preprocessing")
            return None
        
        
    def _preprocess_image(data: np.ndarray, method: str, new_size: int) -> Union[np.ndarray, None]:
        """
        Preprocess either images.

        Parameters
        ----------
        data : np.ndarray
            The data to be preprocessed.
        method : str
            The preprocessing method.
        new_size : int
            The size of the preprocessed data.

        Returns
        -------
        new_data : np.ndarray
            The preprocessed data.
        None
            If the preprocessing failed.
        """
        if method == "resize":
            return SamplePreprocessor._resize_images(data, new_size)
        elif method == "crop":
            return SamplePreprocessor._crop_images(data, new_size)
        else:
            log.error(f"{method} is not a valid image preprocessing method")
            return None

    
    def _preprocess_signal(data: np.ndarray, method: str, new_size: int) -> Union[np.ndarray, None]:
        """
        Preprocess either signals.

        Parameters
        ----------
        data : np.ndarray
            The data to be preprocessed.
        method : str
            The preprocessing method.
        new_size : int
            The size of the preprocessed data.

        Returns
        -------
        new_data : np.ndarray
            The preprocessed data.
        None
            If the preprocessing failed.
        """
        if method == "resample":
            return SamplePreprocessor._resample_signal(data, new_size)
        if method == "crop":
            return SamplePreprocessor._crop_signal(data, new_size)
        else:
            log.error(f"{method} is not a valid signal preprocessing method")
            return None


    def _resize_images(images: np.ndarray, size: int) -> np.ndarray:
        """
        Resize a batch of images with bilinear interpolation. Input images must be of shape (b, c, h, w).

        Parameters
        ----------
        images : np.ndarray
            The original images of shape (b, c, h, w).
        size : int
            The new size.

        Returns
        -------
        images : np.ndarray
            The resized images.
        """
        output_size    = (size, size)
        resized_images = np.empty((images.shape[0], images.shape[1], *output_size))
        resized_images = resized_images.astype(images.dtype)

        for i, image_chw in enumerate(images):
            for c, image_hw in enumerate(image_chw):
                resized_images[i][c] = cv2.resize(
                    src           = image_hw, 
                    dsize         = output_size, 
                    interpolation = cv2.INTER_LINEAR
                )

        return resized_images


    def _crop_images(images: np.ndarray, size: int) -> np.ndarray:
        """
        Center crop a batch of images. Input images must be of shape (b, c, h, w).

        Parameters
        ----------
        images : np.ndarray
            The original images of shape (b, c, h, w).
        size : int
            The new size.

        Returns
        -------
        images : np.ndarray
            The cropped images.
        """
        output_size    = (size, size)
        input_size     = (images.shape[2], images.shape[3])
        resized_images = np.empty((images.shape[0], images.shape[1], *output_size))
        resized_images = resized_images.astype(images.dtype)

        center_y, center_x = input_size[0] // 2, input_size[1] // 2
        start_x = center_x - (output_size[1] // 2)
        start_y = center_y - (output_size[0] // 2)
        end_x   = start_x + output_size[1]
        end_y   = start_y + output_size[0]

        for i, image_chw in enumerate(images):
            for c, image_hw in enumerate(image_chw):
                resized_images[i][c] = image_hw[start_y:end_y, start_x:end_x]

        return resized_images


    def _resample_signal(signals: np.ndarray, size: int) -> np.ndarray:
        """
        Resample signals using 'resample' for scipy. Input must be of shape (b, s).

        Parameters
        ----------
        signals : np.ndarray
            The original signals of shape (b, s).

        size : int
            The new size.
        
        Returns
        ------
        signals : np.ndarray
            The resampled signals.
        """
        return resample(signals, size, axis=1)


    def _crop_signal(signals: np.ndarray, size: int) -> np.ndarray:
        """
        Center crop a batch of signals. Input signals must be of shape (b, s).

        Parameters
        ----------
        signals : np.ndarray
            The original signals of shape (b, s).

        size : int
            The new size.
        
        Returns
        ------
        signals : np.ndarray
            The cropped signals.
        """
        batch_size, original_size = signals.shape

        start_idx = (original_size - size) // 2
        end_idx   = start_idx + size
        return signals[:, start_idx : end_idx]


#===========================================================================================================================
# Dataset Creation
#
class DatasetHelper:
    """
    Helper class for accessing a hdf5 dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the hdf5 dataset.
    data_key : str
        The key of the main data stored in the dataset.
    """
    def __init__(self, dataset_path: str, data_key: str) -> None:
        self.dataset = h5py.File(dataset_path, "r")
        self.data:   np.ndarray = self.dataset[data_key]
        self.labels: np.ndarray = self.dataset["labels"]

        self.data_shape = list(self.data.shape[1:])
        self.size       = self.data.shape[0]


    def collect_samples(self, indices: List, contiguous: bool) -> Tuple:
        """
        Collect samples from the dataset given a set of indices.

        Parameters
        ----------
        indices : List
            The list of indices.
        contiguous : bool
            Is the list of indices ordered.

        Returns
        -------
        data : np.ndarray
            The data.
        labels : np.ndarray
            The labels.
        """
        if contiguous:
            # The indices are ordered, so we can slice into the original dataset
            data   = self.data[indices]
            labels = self.labels[indices]
        else:
            data   = []
            labels = []
            for i in indices:
                data.append(self.data[i])
                labels.append(self.labels[i])
            
            data   = np.stack(data, axis=0)
            labels = np.stack(labels, axis=0)

        return data, labels


    def close(self) -> None:
        """
        Close the dataset file.
        """
        self.dataset.close()


def create_shuffled_dataset(
    dataset_path: str, 
    output_path:  str,
    data_key:     str,
    batch_size:   int
) -> None:
    """
    Create a shuffled version of a dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the base dataset.
    output_path : str
        The output path.
    data_key : str
        The key of the main data stored in the original hdf5 dataset.
    batch_size : int
        The maximum batch size to use when processing the datasets.
    """
    if check_path_exists(dataset_path) is False:
        return
    
    original_dataset = DatasetHelper(dataset_path, data_key)

    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset(
            data_key, 
            shape    = original_dataset.data.shape, 
            dtype    = original_dataset.data.dtype,
        )
        hdf.create_dataset(
            "labels", 
            shape    = original_dataset.labels.shape, 
            dtype    = original_dataset.labels.dtype
        )

    indices = np.arange(original_dataset.size)
    np.random.shuffle(indices)

    batches     = batch_value(original_dataset.size, batch_size)
    current_idx = 0
    start_time  = time.time()

    log.info(f"Shuffling dataset from {dataset_path}")
    with h5py.File(output_path, "a") as hdf:
        new_data   = hdf[data_key]
        new_labels = hdf["labels"]

        for i, batch in enumerate(batches):
            log.info(f"\tProcessing batch {i + 1}/{len(batches)}")

            batch_indices = indices[current_idx : (current_idx + batch)]

            batch_data, batch_labels = original_dataset.collect_samples(batch_indices, contiguous=False)

            new_data[current_idx : (current_idx + batch)]   = batch_data
            new_labels[current_idx : (current_idx + batch)] = batch_labels

            current_idx += batch

    end_time = time.time()
    log.info(f"Finished processing dataset in {(end_time - start_time):.4f}s. Stored in {output_path}")

    original_dataset.close()


def create_raw_dataset(
    data_path:   str, 
    output_path: str, 
    window_size: int, 
    full:        bool = False
) -> None:
    """
    Create a hdf5 dataset from MIT-BIH.

    Parameters
    ----------
    data_path : str
        Path to MIT-BIH database.
    output_path : str
        Path to output hdf5 dataset.
    window_size : int 
        Window size for heartbeat segments.
    full : bool
        Use all of the records.
    """
    check_path_for_dir(output_path, create=True)

    # Initialize datasets in the hdf5 file
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset(
            "segments", 
            shape    = (0, window_size), 
            maxshape = (None, window_size), 
            chunks   = (1, window_size), 
            dtype    = np.float64
        )
        hf.create_dataset(
            "labels", 
            shape    = (0,), 
            maxshape = (None,), 
            chunks   = (1,), 
            dtype    = int
        )

    log.info(f"Processing all records")
    current_idx = 0
    total_start_time = time.time()

    records = ArrhythmiaLabels.records_full if full is True else ArrhythmiaLabels.records

    for record in records:
        start_time = time.time()

        log.info(f"\tProcessing Record {record}")
        record_segments, record_labels = process_record(f"{data_path}/{record}", window_size)
        num_segments = len(record_segments)
        
        log.info(f"\t\tHeartbeats: {num_segments}")
        
        class_dist = get_class_distribution(record_labels)
        log.info(f"\t\tClasses: {class_dist}")

        # Store record
        with h5py.File(output_path, "a") as hf:
            segments = hf["segments"]
            labels   = hf["labels"]

            segments.resize(segments.shape[0] + num_segments, axis=0)
            labels.resize(labels.shape[0] + num_segments, axis=0)

            segments[current_idx : current_idx + num_segments] = record_segments
            labels[current_idx : current_idx + num_segments]   = record_labels

        current_idx += num_segments
        end_time = time.time()

        log.info(f"\tRecord {record} added to dataset {output_path} in {(end_time - start_time):.4f}s")

    total_end_time = time.time()
    log.info(f"All records have been processed and stored in {output_path}. Total time: {(total_end_time - total_start_time):.4f}s")


def create_image_dataset(signal_dataset_path: str, output_path: str, batch_size: int, preprocess: str = None, new_size: int = 0) -> None:
    """
    Create an image dataset from a previous dataset of 1D signal data. Images are processed in batches, thus make sure you tune this parameter
    such that you do not run out of memory. 
    
    DEPRECATED: Use create_image_dataset_contiguous instead.

    Parameters
    ----------
    signal_dataset_path : str
        Path to hdf5 file containing 1D heartbeat segments.
    output_path : str 
        Path of output hdf5 file.
    batch_size : int
        Number of samples to process at a time. 
    preprocess : str
        Preprocessing method to perform on the images.
    new_size : int
        The new size of the image after preprocessing.
    """
    log.warning("'create_image_dataset' is deprecated. Use create_image_dataset_contiguous instead")
    create_image_dataset_contiguous(signal_dataset_path, output_path, batch_size, preprocess, new_size)
    return

    if not os.path.exists(signal_dataset_path):
        log.error(f"{signal_dataset_path} does not exist")
        return

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    signal_dataset = h5py.File(signal_dataset_path, "r")
    segments       = signal_dataset["segments"]
    labels         = signal_dataset["labels"]

    image_size  = segments.shape[-1] if preprocess is None else new_size
    num_samples = len(segments)

    # Initialize dataset for images and labels
    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset(
            "images", 
            shape    = (0, 3, image_size, image_size), 
            maxshape = (None, 3, image_size, image_size), 
            chunks   = (1, 3, image_size, image_size), 
            dtype    = np.float32
        )
        hdf.create_dataset(
            "labels", 
            shape    = (0,), 
            maxshape = (None,), 
            chunks   = (1,), 
            dtype    = int
        )
        
    # The labels should be the exact same
    with h5py.File(output_path, "a") as hdf:
        image_labels = hdf["labels"]
        image_labels.resize(image_labels.shape[0] + len(labels), axis=0)
        image_labels[:len(labels)] = labels

    # Transform samples in batches
    batches = batch_value(num_samples, batch_size)
    current_idx = 0
    total_start_time = time.time()

    log.info(f"Transforming signals into images from {signal_dataset_path}")
    log.info(f"\tTotal:      {num_samples}")
    log.info(f"\tPreprocess: {preprocess}")
    log.info(f"\tImage Size: {image_size}")

    for i, batch in enumerate(batches):
        log.info(f"\tProcessing batch {i + 1}/{len(batches)}")
        start_time = time.time()

        batch_images = signal_to_image(segments[current_idx : (current_idx + batch)])

        if preprocess is not None:
            if preprocess == "resize":
                batch_images = resize_images(batch_images, new_size)
            elif preprocess == "crop":
                batch_images = crop_images(batch_images, new_size)

        # Store the images
        with h5py.File(output_path, "a") as hdf:
            images = hdf["images"]
            images.resize(images.shape[0] + batch, axis=0)

            images[current_idx : (current_idx + batch)] = batch_images

        end_time = time.time()
        log.info(f"\tCompleted batch in {(end_time - start_time):.4f}s")

        current_idx += batch

    total_end_time = time.time()
    log.info(f"Image transformation complete. Total time: {(total_end_time - total_start_time):4f}s")

    signal_dataset.close()


def create_image_dataset_contiguous(signal_dataset_path: str, output_path: str, batch_size: int, preprocess: str = None, new_size: int = 0) -> None:
    """
    Create an image dataset from a previous dataset of 1D signal data. Images are processed in batches, thus make sure you tune this parameter
    such that you do not run out of memory. This version does not use hdf5 chunking.

    Parameters
    ----------
    signal_dataset_path : str
        Path to hdf5 file containing 1D heartbeat segments.
    output_path : str 
        Path of output hdf5 file.
    batch_size : int
        Number of samples to process at a time. 
    preprocess : str
        Preprocessing method to perform on the images.
    new_size : int
        The new size of the image after preprocessing.
    """
    if check_path_exists(signal_dataset_path) is False:
        return
    check_path_for_dir(output_path, create=True)

    signal_dataset = DatasetHelper(signal_dataset_path, "segments")
    image_size     = signal_dataset.data_shape[0] if preprocess is None else new_size

    # Initialize dataset for images and labels
    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset(
            "images", 
            shape    = (signal_dataset.size, 3, image_size, image_size), 
            dtype    = np.float32,
        )
        hdf.create_dataset(
            "labels", 
            shape    = (signal_dataset.size,), 
            dtype    = int
        )
        
        # The labels should be the exact same
        image_labels = hdf["labels"]
        image_labels[:signal_dataset.size] = signal_dataset.labels

    # Transform samples in batches
    batches = batch_value(signal_dataset.size, batch_size)
    current_idx = 0
    total_start_time = time.time()

    log.info(f"Transforming signals into images from {signal_dataset_path}")
    log.info(f"\tTotal:      {signal_dataset.size}")
    log.info(f"\tPreprocess: {preprocess}")
    log.info(f"\tImage Size: {image_size}")
    
    log.info(f"\tBegin transformation...")
    with h5py.File(output_path, "a") as hdf:
        images = hdf["images"]

        for i, batch in enumerate(batches):
            log.info(f"\t\tProcessing batch {i + 1}/{len(batches)}")
            start_time = time.time()

            batch_images = signal_to_image(signal_dataset.data[current_idx : (current_idx + batch)])
            if preprocess is not None:
                batch_images = SamplePreprocessor.preprocess(batch_images, preprocess, "image", new_size)

            images[current_idx : (current_idx + batch)] = batch_images

            end_time = time.time()
            log.info(f"\t\tCompleted batch in {(end_time - start_time):.4f}s")

            current_idx += batch

    total_end_time = time.time()
    log.info(f"Image transformation complete. Total time: {(total_end_time - total_start_time):4f}s")

    signal_dataset.close()


def create_train_test_from_dataset(
    dataset_path: str, 
    output_name:  str, 
    ratio:        float, 
    data_key:     str, 
    shuffle:      bool, 
    batch_size:   int
) -> None:
    """
    Create train and test datasets from a base dataset. This will result in two new hdf5 files.

    Parameters
    ----------
    dataset_path : str
        The path to the base hdf5 dataset to split.
    output_name : str
        The output file name for the split datasets. Do not include the extension. '_train' and '_test' will be appended
        to the end of the respective files.
    ratio : float
        The train-test split ratio.
    data_key : str
        The key of the main data stored in the original hdf5 dataset.
    shuffle : bool
        Whether or not to shuffle the dataset before splitting.
    batch_size : int
        The maximum batch size to use when processing the split datasets.
    """
    if check_path_exists(dataset_path) is False:
        return
    dataset_dir = os.path.dirname(dataset_path)
    
    original_dataset = DatasetHelper(dataset_path, data_key)

    train_indices, test_indices = split_value(original_dataset.size, ratio, shuffle)

    log.info(f"Generating split datasets")
    log.info(f"\tShuffle: {shuffle}")
    log.info(f"\tTotal:   {original_dataset.size}")
    log.info(f"\tTrain:   {len(train_indices)} ({(len(train_indices) / original_dataset.size * 100):.2f}%)")
    log.info(f"\tTest:    {len(test_indices)} ({(len(test_indices) / original_dataset.size * 100):.2f}%)")

    splits = {
        "train": train_indices,
        "test":  test_indices
    }
    for split, indices in splits.items():
        log.info(f"Processing split - {split}")
        split_path = f"{dataset_dir}/{output_name}_{split}.hdf5"
        start_split_time = time.time()

        log.info("\tInitializing dataset")
        with h5py.File(split_path, "w") as hdf:
            hdf.create_dataset(
                data_key, 
                shape    = tuple([len(indices)] + original_dataset.data_shape), 
                dtype    = original_dataset.data.dtype
            )
            hdf.create_dataset(
                "labels", 
                shape    = (len(indices),), 
                dtype    = original_dataset.labels.dtype
            )

        # Copy over data
        log.info("\tCollecting data from original dataset")
        batches     = batch_value(len(indices), batch_size)
        current_idx = 0

        with h5py.File(split_path, "a") as hdf:
            split_data   = hdf[data_key]
            split_labels = hdf["labels"]

            for i, batch in enumerate(batches):
                log.info(f"\t\tProcessing batch {i + 1}/{len(batches)}")
                batch_indices = indices[current_idx : (current_idx + batch)]

                batch_data, batch_labels = original_dataset.collect_samples(batch_indices, contiguous=False)

                split_data[current_idx : (current_idx + batch)]   = batch_data
                split_labels[current_idx : (current_idx + batch)] = batch_labels

                current_idx += batch

        end_split_time = time.time()
        log.info(f"Finished processing {split} split in {(end_split_time - start_split_time):.4f}s. Stored in {split_path}")

    log.info(f"Finished generating dataset splits")

    original_dataset.close()


def create_subset_from_dataset(
    dataset_path: str, 
    output_path:  str,
    data_key:     str,
    method:       str,
    ratio:        float,
    shuffle:      bool,
    batch_size:   int
) -> None:
    """
    Create a subset of an original dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the base hdf5 dataset.
    output_path : str
        The output path for the subset.
    data_key : str
        The key of the main data stored in the original hdf5 dataset.
    method : str
        The subset creation method, either 'balance' or 'random'.
    ratio : float
        The subset ratio.
    shuffle : bool
        Whether or not to shuffle the dataset before creating the subset.
    batch_size : int
        The maximum batch size to use when processing the datasets.
    """
    if check_path_exists(dataset_path) is False:
        return

    check_path_for_dir(output_path, create=True)

    original_dataset = DatasetHelper(dataset_path, data_key)

    # Collect indices given the method
    if method == "random":
        indices = random_subset(original_dataset.size, ratio, shuffle)
    elif method == "balance":
        MINIMUM_NORMALS = 5000
        indices = remove_normals(original_dataset.labels[:], ratio, MINIMUM_NORMALS)
        if indices is None:
            log.error("Failed to create subset")
            return
    else:
        log.error(f"{method} is an invalid method. Use either 'random' or 'balance'")
        return

    log.info(f"Generating subset from {dataset_path}")
    log.info(f"\tMethod: {method}")
    log.info(f"\tTotal:  {original_dataset.size}")
    log.info(f"\tSubset: {len(indices)} ({(len(indices) / original_dataset.size * 100):.2f}%)")

    # Initialize the new dataset
    log.info("\tInitializing dataset")
    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset(
            data_key, 
            shape    = tuple([len(indices)] + original_dataset.data_shape), 
            dtype    = original_dataset.data.dtype
        )
        hdf.create_dataset(
            "labels", 
            shape    = (len(indices),), 
            dtype    = original_dataset.labels.dtype
        )

    # Copy over data
    log.info("\tCollecting data from original dataset")
    batches     = batch_value(len(indices), batch_size)
    current_idx = 0

    start_time = time.time()
    with h5py.File(output_path, "a") as hdf:
        subset_data   = hdf[data_key]
        subset_labels = hdf["labels"]

        for i, batch in enumerate(batches):
            log.info(f"\t\tProcessing batch {i + 1}/{len(batches)}")
            batch_indices = indices[current_idx : (current_idx + batch)]

            is_contiguous = (shuffle is False and method == "random")
            batch_data, batch_labels = original_dataset.collect_samples(batch_indices, is_contiguous)

            subset_data[current_idx : (current_idx + batch)]   = batch_data
            subset_labels[current_idx : (current_idx + batch)] = batch_labels

            current_idx += batch

    end_time = time.time()
    log.info(f"Finished processing subset in {(end_time - start_time):.4f}s. Stored in {output_path}")

    original_dataset.close()


def create_preprocessed_dataset(
    dataset_path: str, 
    output_path:  str,
    data_key:     str,
    data_type:    str,
    method:       str,
    new_size:     int,
    batch_size:   int
) -> None:
    """
    Create a preprocessed version of a dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the original dataset.
    output_path : str
        The path to the output dataset.
    data_key : str
        The key of the main data stored in the original hdf5 dataset.
    data_type : str
        The type of input, either 'image' or 'signal'.
    method : str
        The preprocessing method, either 'resize', 'crop', or 'resample'.
    new_size : int
        The new size after preprocessing.
    batch_size : int
        The maximum batch size to use when processing the dataset.
    """
    if check_path_exists(dataset_path) is False:
        return
    
    check_path_for_dir(output_path, create=True)

    original_dataset = DatasetHelper(dataset_path, data_key)

    if data_type == "image":
        new_data_shape = [original_dataset.data_shape[0], new_size, new_size]
    elif data_type == "signal":
        new_data_shape = [new_size]
    else:
        log.error(f"'{data_type}' is not supported")
        return
        
    log.info(f"Generating preprocessed dataset from {dataset_path}")
    log.info(f"\tMethod:   {method}")
    log.info(f"\tTotal:    {original_dataset.size}")
    log.info(f"\tOriginal: {tuple(original_dataset.data_shape)}")
    log.info(f"\tNew:      {tuple(new_data_shape)}")

    # Initialize the new dataset
    log.info("\tInitializing dataset")
    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset(
            data_key, 
            shape    = tuple([original_dataset.size] + new_data_shape), 
            dtype    = original_dataset.data.dtype
        )
        hdf.create_dataset(
            "labels", 
            shape    = (original_dataset.size,), 
            dtype    = original_dataset.labels.dtype
        )

    # Copy over data
    log.info("\tPreprocessing data from original dataset")
    batches     = batch_value(original_dataset.size, batch_size)
    current_idx = 0

    start_time = time.time()
    with h5py.File(output_path, "a") as hdf:
        new_data   = hdf[data_key]
        new_labels = hdf["labels"]

        for i, batch in enumerate(batches):
            log.info(f"\t\tProcessing batch {i + 1}/{len(batches)}")
            batch_data, batch_labels = original_dataset.collect_samples(range(current_idx, current_idx + batch), contiguous=True)

            new_batch_data = SamplePreprocessor.preprocess(batch_data, method, data_type, new_size)
            if new_batch_data is None:
                original_dataset.close()
                return

            new_data[current_idx : (current_idx + batch)]   = new_batch_data
            new_labels[current_idx : (current_idx + batch)] = batch_labels

            current_idx += batch

    end_time = time.time()
    log.info(f"Finished processing dataset in {(end_time - start_time):.4f}s. Stored in {output_path}")

    original_dataset.close()


def create_full_split_sets(
    dataset_path: str, 
    output_dir:   str,
    data_key:     str,
    num_sets:     int,
    shuffle:      bool,
    batch_size:   int
) -> None:
    """
    Create subsets from an original dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the base hdf5 dataset.
    output_path : str
        The output path for the subset.
    data_key : str
        The key of the main data stored in the original hdf5 dataset.
    num_sets : int
        The number of subsets.
    shuffle : bool
        Whether or not to shuffle the dataset before creating the subsets.
    batch_size : int
        The maximum batch size to use when processing the datasets.
    """
    if check_path_exists(dataset_path) is False:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_dataset = DatasetHelper(dataset_path, data_key)

    # Get set indices
    full_indices = random_subset(original_dataset.size, 1.0, shuffle=shuffle)
    indices      = np.array_split(full_indices, num_sets)

    log.info(f"Generating full split datasets from {dataset_path}")
    log.info(f"\tTotal:    {original_dataset.size}")
    log.info(f"\tNum Sets: {num_sets}")
    for i, s in enumerate(indices):
        log.info(f"\tSet {i}:    {len(s)}")

    total_start_time = time.time()
    for i, set_indices in enumerate(indices):
        set_path = f"{output_dir}/set_{i}.hdf5"
        log.info(f"\tGenerating set {i} and saving to {set_path}")

        # Initialize datasets
        log.info(f"\t\tInitializing dataset")
        with h5py.File(set_path, "w") as hdf:
            hdf.create_dataset(
                data_key, 
                shape    = tuple([len(set_indices)] + original_dataset.data_shape), 
                dtype    = original_dataset.data.dtype
            )
            hdf.create_dataset(
                "labels", 
                shape    = (len(set_indices),), 
                dtype    = original_dataset.labels.dtype
            )

        batches     = batch_value(len(set_indices), batch_size)
        current_idx = 0

        # Copy over data
        log.info("\t\tCollecting data from original dataset")
        start_time = time.time()
        with h5py.File(set_path, "a") as hdf:
            subset_data   = hdf[data_key]
            subset_labels = hdf["labels"]

            for j, batch in enumerate(batches):
                log.info(f"\t\t\tProcessing batch {j + 1}/{len(batches)}")
                batch_indices = set_indices[current_idx : (current_idx + batch)]

                batch_data, batch_labels = original_dataset.collect_samples(batch_indices, contiguous=False)

                subset_data[current_idx : (current_idx + batch)]   = batch_data
                subset_labels[current_idx : (current_idx + batch)] = batch_labels

                current_idx += batch
        
        end_time = time.time()
        log.info(f"\t\tFinished processing subset {i} in {(end_time - start_time):.4f}s. Stored in {set_path}")

    total_end_time = time.time()
    log.info(f"Finished processing all {num_sets} sets in {(total_end_time - total_start_time):.4f}s")

    original_dataset.close()


#===========================================================================================================================
# Main (Used for testing this file)
#
if __name__ == "__main__":
    """
    Main used for testing.
    """
    d1 = DatasetHelper("../Data/MIT-BIH-Raw/Datasets/Resolution-64/signal_full.hdf5", "segments")
    d2 = DatasetHelper("../Data/MIT-BIH-Raw/Datasets/Resolution-64/signal_full_shuffled.hdf5", "segments")

    log.info(f"HELP: {d1.data_shape}, {d1.size}")
    log.info(f"HELP: {d2.data_shape}, {d2.size}")

    total_same = 0
    for i in range(d1.size):
        d1_data = d1.data[i]
        d2_data = d2.data[i]

        if np.array_equal(d1_data, d2_data):
            total_same += 1

    log.info(f"HELP: {total_same}")

    d1.close()
    d2.close()