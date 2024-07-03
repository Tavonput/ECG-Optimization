import h5py
import wfdb
import logging
import os
import time

import numpy as np

from scipy.signal import butter, filtfilt
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField


LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("DTGEN").setLevel(logging.INFO)
log = logging.getLogger("DTGEN")


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

    
def segment_heartbeat(signal: np.ndarray, peak: int, window_size: int) -> np.ndarray | None:
    """
    Segment a heartbeat. The length of the segment will be 2 x window_size.

    @param signal: The ECG signal.
    @param peak: The r-peak of the heartbeat.
    @param window_size: The window size.

    @return The heartbeat segment or None if the segment is outside of the ECG signal.
    """
    start_idx = peak - window_size
    end_idx   = peak + window_size
    
    if start_idx >= 0 and end_idx < len(signal):
        return signal[start_idx : end_idx]
    
    return None


def process_record(record_path: str, window_size: int) -> tuple[list, list]:
    """
    Process a record given a window size.

    @param record_path: Path to the record.
    @param window_size: The window size (total size).

    @return Array of heartbeat segments and an array of labels.
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


def get_class_distribution(labels: list) -> dict[str, int]:
    """
    Get the class distribution of a list of labels.

    @param labels: List of labels.

    @return Class distribution in the form of a dictionary (class name : count).
    """
    counts = dict()
    for i, label in enumerate(ArrhythmiaLabels.classes):
        counts[label] = np.sum(labels == i)

    return counts


def examine_dataset(dataset_path: str) -> dict[str, int]:
    """
    Examine the keys in a hdf5 file.

    @param dataset_path: Path to hdf5 file.
    
    @return Dataset information in the form of a dictionary.
    """
    data = dict()
    with h5py.File(dataset_path, "r") as hdf:
        for key in hdf.keys():
            data[f"{key}_size"] = len(hdf[key])
            data[f"{key}_shape"] = hdf[key].shape
            data[f"{key}_dtype"] = hdf[key].dtype
            
    return data


def create_raw_dataset(data_path: str, output_path: str, window_size: int) -> None:
    """
    Create a hdf5 dataset from MIT-BIH.

    @param data_path: Path to MIT-BIH database.
    @param output_path: Path to output hdf5 dataset.
    @param window_size: Window size for heartbeat segments.
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    for record in ArrhythmiaLabels.records:
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


def batch_value(value: int, batch_size: int) -> list[int]:
    """
    Calculate batching. For example if value is 33 and the batch size is 10, the corresponding batching will be [10, 10, 10, 3].

    @param value: Number of samples to batch.
    @param batch_size: The batch_size.

    @return Batching list.
    """
    full_batches = [batch_size] * (value // batch_size)
    remainder    = value % batch_size

    if remainder != 0:
        full_batches += [remainder]
        
    return full_batches


def signal_to_image(signals: np.ndarray) -> np.ndarray:
    """
    Transform signals into images. Signals should be of shape (num_samples, image_size).

    @param signals: The 1D signals to transform.

    @return The transformed images of shape (num_samples, 3, image_size, image_size).
    """
    gaf = GramianAngularField(method="summation")
    gaf_image = gaf.fit_transform(signals)

    rp = RecurrencePlot(threshold="point")
    rp_image = rp.fit_transform(signals)
    
    mtf = MarkovTransitionField()
    mtf_image = mtf.fit_transform(signals)

    images = np.stack((gaf_image, rp_image, mtf_image), axis=0) # (3, batch_size, width, height)
    return images.transpose(1, 0, 2, 3) # (batch_size, 3, width, height)


def create_image_dataset(signal_dataset_path: str, output_path: str, batch_size: int) -> None:
    """
    Create an image dataset from a previous dataset of 1D signal data. Images are processed in batches, thus make sure you tune this parameter
    such that you do not run out of memory.

    @param signal_dataset_path: Path to hdf5 file containing 1D heartbeat segments.
    @param output_path: Path of output hdf5 file.
    @param batch_size: Number of samples to process at a time. 
    """
    if not os.path.exists(signal_dataset_path):
        log.error(f"{signal_dataset_path} does not exist")
        return

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    signal_dataset = h5py.File(signal_dataset_path, "r")
    segments       = signal_dataset["segments"]
    labels         = signal_dataset["labels"]

    image_size = segments.shape[-1]
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

    log.info("Transforming signals into images")
    for i, batch in enumerate(batches):
        log.info(f"\tProcessing batch {i + 1} / {len(batches) + 1}")
        start_time = time.time()

        batch_images = signal_to_image(segments[current_idx : (current_idx + batch)])
        
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
