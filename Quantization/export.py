import os

import torch

def export_to_onnx(model, image_size, batch_size, export_path) -> None:
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

    print(f"[INFO]: Exporting to ONNX")
    torch.onnx.export(
        model, 
        dummy_input, 
        export_path,
        opset_version       = 18,
        verbose             = False,
        input_names         = ["input"],
        output_names        = ["output"],
        do_constant_folding = True,
        # dynamic_axes        = dynamic,
    )

    print(f"[INFO]: Exported to {export_path}")