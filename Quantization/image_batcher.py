import h5py
import logging
import time

import numpy as np
import cv2

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("IMGBT").setLevel(logging.DEBUG)
log = logging.getLogger("IMGBT")


#===========================================================================================================================
# Image Batcher
#
class ImageBatcher:
    """
    Image Batcher

    Batches images from a hdf5. An instance of an ImageBatcher should be used as an iterator. For quickest batching, the
    dataset must be pre-shuffled. Samples from an hdf5 file must be loaded one at a time if using random access (needed for 
    shuffling), which can be quite slow.

    Parameters
    ----------
    file_path : str
        Path to hdf5 file.
    batch_size : int
        Batch size.
    image_sie : int
        Image size.
    shuffle : bool
        Whether or not to shuffle the data.
    drop_last : bool
        Whether or not to drop the last batch if it is not equal to the batch size.
    max_batch : int
        Maximum number of batches to load from dataset. A max batch of 0 (default) means no maximum.
    """
    def __init__(
        self, 
        file_path:  str, 
        batch_size: int, 
        image_size: int,
        shuffle:    bool,
        drop_last:  bool,
        max_batch:  int = 0
    ) -> None:
        log.info("Initializing image batcher")
        self.file_path  = file_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.max_batch  = max_batch

        log.info(f"Loading dataset from {self.file_path}")
        self.h5_file    = h5py.File(self.file_path)
        self.images     = self.h5_file["images"]
        self.num_images = self.images.shape[0]

        self.shape  = (batch_size, 3, image_size, image_size)
        self.size   = int(np.prod(self.shape))
        self.dtype  = self.images.dtype
        self.nbytes = self.dtype.itemsize * self.size

        log.debug(f"Image Fields... \
            \n\tNumber of images: {self.num_images} \
            \n\tType: {self.dtype} \
            \n\tBatch Shape: {self.shape} \
            \n\tBatch Size: {self.size} \
            \n\tBatch Bytes: {self.nbytes}"
        )

        self.indices       = None
        self.current_index = 0
        self.current_batch = 0
        self.epoch         = 0
        self._initialize_indices()

        log.info("Successfully created image batcher")


    def __del__(self):
        self.h5_file.close()


    def _initialize_indices(self) -> None:
        """
        Initialize indices and shuffle if needed.
        """
        log.debug(f"Initializing indices. Shuffle: {self.shuffle}")
        self.indices = np.arange(self.num_images)

        if self.shuffle:
            np.random.shuffle(self.indices)


    def __iter__(self):
        return self


    def __next__(self) -> np.ndarray:
        """
        Load the next batch.

        Returns
        -------
        batch : np.ndarray
            Batch of images.
        """
        self.current_batch += 1

        if self.current_index >= self.num_images:
            self._batch_completed()
            raise StopIteration
        
        if self.max_batch != 0 and self.current_batch > self.max_batch:
            self._batch_completed()
            raise StopIteration
        
        start_idx     = self.current_index
        end_idx       = min(self.current_index + self.batch_size, self.num_images)
        batch_indices = self.indices[start_idx : end_idx]

        if self.drop_last and end_idx - start_idx != self.batch_size:
            log.debug("Skipped last batch")
            self._batch_completed()
            raise StopIteration
        
        batch_images = self._get_batch_from_dataset(batch_indices)

        self.current_index = end_idx
        return batch_images
    

    def _batch_completed(self) -> None:
        log.info("Batch completed")
        self.epoch         += 1
        self.current_index  = 0
        self.current_batch  = 0

        if self.shuffle:
            np.random.shuffle(self.indices)
    

    def _get_batch_from_dataset(self, indices: list[int]) -> np.ndarray:
        """
        Get the batch of images from the dataset given the indices. Handles shuffling.

        Parameters
        ----------
        indices : list[int]
            List of indices.

        Returns
        -------
        batch : np.ndarray
            Batch as a numpy array.
        """
        if self.shuffle:
            batch_images = []
            for i in indices:
                batch_images.append(self.images[i])
            
            batch_images = np.stack(batch_images, axis=0)
            return self._preprocess_images(batch_images)
        else:
            batch_images = self.images[indices]
            return self._preprocess_images(batch_images)


    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images. Performs a resize with bilinear interpolation. Images must be of shape (b, c, h, w).

        Parameters
        ----------
        images : np.ndarray
            Batch of images (b, c, h, w).

        Returns
        -------
        batch : np.ndarray
            Preprocessed batch.
        """
        output_size    = (self.image_size, self.image_size)
        resized_images = np.empty((images.shape[0], images.shape[1], *output_size))
        resized_images = resized_images.astype(self.dtype)

        for i, image_chw in enumerate(images):
            for c, image_hw in enumerate(image_chw):
                resized_images[i][c] = cv2.resize(
                    src           = image_hw, 
                    dsize         = (self.image_size, self.image_size), 
                    interpolation = cv2.INTER_LINEAR
                )
        
        return resized_images


def benchmark_batch_loading(batcher: ImageBatcher, num_batches: int = 100) -> float:
    start_time = time.time()

    for i, batch in enumerate(batcher):
        if i >= num_batches - 1:
            break

    end_time = time.time()
    return (end_time - start_time) / num_batches


#===========================================================================================================================
# Main (Used for testing this file)
#
if __name__ == "__main__":
    """
    Main function can be used for testing.
    """
    batcher = ImageBatcher(
        file_path  = "Data/mitbih_mif_train_small.h5",
        batch_size = 32,
        image_size = 152,
        shuffle    = False,
        drop_last  = True
    )

    throughput = benchmark_batch_loading(batcher, num_batches=1)
    log.info(f"Average time to load a batch: {round(throughput * 1000, 2)}ms")
