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
    Initialize quantization with pytorch_quantization.
    """
    log.info("Initializing quantization with pytorch_quantization")
    quant_modules.initialize()
    log.info("Ready to load models")


def set_input_calibrators(method: str, bits: int = 8) -> None:
    """
    Set the method and bit width of the inputs.

    Parameters
    ----------
    method : str
        The calibration method. Either 'max' or 'histogram'.
    bits : int
        The bit width.
    """
    if method == "max":
        log.info(f"Setting input calibrators to 'max' with bit width {bits}")
        quant_desc = QuantDescriptor(num_bits=bits) # Default is max
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
        return
    
    elif method == "histogram":
        log.info(f"Setting input calibrators to 'histogram' with bit width {bits}")
        quant_desc = QuantDescriptor(calib_method="histogram", num_bits=bits)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc)
        return
    
    else:
        log.warn(f"'{method}' is not supported. Falling back to 'max'")
        return


def set_weight_calibrators(method: str, bits: int = 8) -> None:
    """
    Set the method and bit width of the weights.

    Parameters
    ----------
    method : str
        The calibration method. Either 'max' or 'histogram'.
    bits : int
        The bit width.
    """
    if method == "max":
        log.info(f"Setting weight calibrators to 'max' with bit width {bits}")
        quant_desc = QuantDescriptor(num_bits=bits) # Default is max
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc)
        return
    
    elif method == "histogram":
        log.info(f"Setting input calibrators to 'histogram' with bit width {bits}")
        quant_desc = QuantDescriptor(calib_method="histogram", num_bits=bits)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc)
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
    Calibrate a model.

    Parameters
    ----------
    model : nn.Module
        The model.
    dataloader : DataLoader
        The dataloader to use for calibration.
    num_batches : int
        The number of calibration batches.
    method : str
        The calibration method.
    **kwargs
        Passed to the given calibration method.
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

    Parameters
    ----------
    model : nn.Module
        Model to export.
    image_size : int
        Image size.
    batch_size : int
        Batch size.
    export_path : str
        Path to save the ONNX model.
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
            verbose             = False,
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
    Collect statistics for calibration.
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
    Compute the calibration scales.
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
