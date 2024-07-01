import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from tqdm.auto import tqdm

import logging
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.getLogger("EXQ").setLevel(logging.INFO)
log = logging.getLogger("EXQ")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def quant_initialize() -> None:
    """
    """
    log.info("Initializing quantization with pytorch_quantization")
    quant_modules.initialize()
    log.info("Ready to load models")


def set_input_calibrators(method: str) -> None:
    """
    """
    if method == "max":
        log.info("Setting input calibrators to 'max'")
        return # Default is max
    
    elif method == "histogram":
        log.info("Setting input calibrators to 'histogram'")
        quant_desc = QuantDescriptor(calib_method="histogram")
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
        return
    
    else:
        log.warn(f"'{method}' is not supported. Falling back to 'max'")
        return


def calibrate(
    model:       nn.Module, 
    dataloader:  DataLoader, 
    num_batches: int, 
    method:      str, 
    **kwargs
) -> None:
    """
    """
    with torch.no_grad():
        _collect_stats(model, dataloader, num_batches)
        _compute_amax(model, method=method, **kwargs)

    log.info("Calibration finished. Model is ready for PTQ or to be finetuned with QAT")


def export_to_onnx(
    model:       nn.Module, 
    image_size:  int,
    batch_size:  int, 
    export_path: str
) -> None:
    """
    Export a QAT model to ONNX.

    @param model: Model to export.
    @param image_size: Image size.
    @param batch_size: Batch size.
    @param export_path: Path to save the ONNX model.
    """
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    onnx_dir = os.path.dirname(export_path)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)

    log.info(f"Exporting model to ONNX")
    with pytorch_quantization.enable_onnx_export():
        torch.onnx.export(
            model, 
            dummy_input, 
            export_path,
            opset_version       = 17,
            verbose             = True,
            input_names         = ["input"],
            output_names        = ["output"],
            do_constant_folding = True,
        )

    log.info(f"Exporting finished. ONNX model saved to {export_path}")


#===========================================================================================================================
# Private Helpers
#
# https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html#
def _collect_stats(model, data_loader, num_batches):
    """
    """
    log.info("Collecting information from activations")

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Do calibration
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.to(device))
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def _compute_amax(model, **kwargs):
    """
    """
    log.info("Computing scale for calibrators")

    # Load calibration results
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.to(device)
