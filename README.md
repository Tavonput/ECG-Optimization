# ECG Optimization
Optimized deep learning for Arrhythmia classification using the MIT BIH database.

## Repository Structure
`Experiments` contains code for running experiments.

`Notebooks` contains the main implementations for running experiments.

`Dataset` contains everything related to data generation and working with the dataset.

`Quantization` contains everything related to quantization.

`Scripts` contains useful scripts for exporting to ONNX and building a TensorRT engine.

`Tests` contains unit tests.

`Utils` contains useful utilities, for example loading models from torchvision.

## Quantization With TensorRT
This repository uses TensorRT 10.2. Earlier versions of TensorRT may not work.

## Notes
Some dependencies in the `requirements.txt` will not download automatically with just pip. For example, *pycuda* requires that you have the cuda toolkit properly installed on your system.
*pytorch_quantization* might require manual installation.
