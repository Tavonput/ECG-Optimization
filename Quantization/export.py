import os

import torch


def export_to_onnx(
    model:       torch.nn.Module, 
    image_size:  int, 
    batch_size:  int, 
    export_path: str
) -> None:
    """
    Export a PyTorch model to ONNX.

    @param model: Model to export.
    @param image_size: Image size.
    @param batch_size: Batch size.
    @param export_path: Path to save the ONNX model.
    """
    # Setup dummy input
    dummy_input = torch.randn((batch_size, 3, image_size, image_size))

    # Export to ONNX
    onnx_dir = os.path.dirname(export_path)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)

    # dynamic = {
    #     "input":  {0: "batch_size"},
    #     "output": {0: "batch_size"}
    # }

    print(f"[I]: Exporting to ONNX")
    torch.onnx.export(
        model, 
        dummy_input, 
        export_path,
        opset_version       = 17,
        verbose             = False,
        input_names         = ["input"],
        output_names        = ["output"],
        do_constant_folding = True,
        # dynamic_axes        = dynamic,
    )

    print(f"[I]: Exported to {export_path}")