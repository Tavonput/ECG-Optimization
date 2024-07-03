import argparse
import sys
sys.path.append("../")

import torch
from Quantization.export import export_to_onnx
from Utils.model_loading import load_model_from_pretrained, load_vgg_custom_from_pruned
from Dataset.data_generation import ArrhythmiaLabels

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", 
        type    = str, 
        default = None, 
        help    = "Name of model"
    )
    parser.add_argument(
        "--model_path", 
        type    = str, 
        default = None, 
        help    = "Path to model"
    )
    parser.add_argument(
        "--onnx",       
        type    = str, 
        default = "./", 
        help    = "Path to save onnx model"
    )
    parser.add_argument(
        "--image_size", 
        type    = int, 
        default = 152,  
        help    = "Image size"
    )
    parser.add_argument(
        "--batch_size", 
        type    = int, 
        default = 1,    
        help    = "Batch size"
    )
    parser.add_argument(
        "--pruned_vgg", 
        type    = float, 
        default = 0.0, 
        help    = "VGG layer pruning ratio"
    )
    
    args = parser.parse_args()

    if not args.model_name or not args.model_path:
        print("[ERROR]: Model not provided correctly")
        exit()

    return args
    
#===========================================================================================================================
# Main
#
def main():
    args = parse_args()
    if args.pruned_vgg != 0.0:
        model = load_vgg_custom_from_pruned(args.model_path, args.pruned_vgg, torch.rand((1, 3, args.image_size, args.image_size)), ArrhythmiaLabels.size)
    else:
        model = load_model_from_pretrained(args.model_name, args.model_path, ArrhythmiaLabels.size)
    export_to_onnx(model, args.image_size, args.batch_size, args.onnx)

if __name__ == "__main__":
    main()