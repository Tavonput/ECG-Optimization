import argparse

from Quantization.build_engine import build_engine, CalibrationConfig

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx", 
        type    = str, 
        default = None, 
        help    = "Path to ONNX file"    
    )
    parser.add_argument(
        "--engine", 
        type    = str, 
        default = "./engine.engine", 
        help    = "Path to save engine"
    )
    parser.add_argument(
        "--precision",
        type    = str,
        default = "fp16",
        choices = ["fp32", "fp16", "int8"],
        help    = " Model precision: 'fp32', 'fp16', or 'int8'"
    )
    parser.add_argument(
        "--calib_path",
        type    = str,
        default = None,
        help    = "Path to calibration dataset"
    )
    parser.add_argument(
        "--calib_cache",
        type    = str,
        default = None,
        help    = "Path to calibration cache, either to save or to load"
    )
    parser.add_argument(
        "--calib_batch_size",
        type    = int,
        default = 32,
        help    = "Batch size for calibration",
    )
    parser.add_argument(
        "--calib_max_batch",
        type    = int,
        default = 100,
        help    = "Maximum number of batches to use for calibration",
    )
    parser.add_argument(
        "--calib_image_size",
        type    = int,
        default = 152,
        help    = "Image size for calibration",
    )

    args = parser.parse_args()

    if not args.onnx:
        print("[ERROR] ONNX model not provided")
        exit(1)

    if args.precision == "int8" and not any([args.calib_path, args.calib_cache]):
        parser.print_help()
        print("[ERROR] INT8 precision requires either --calib_path or --calib_cache")
        exit(1)

    return args

#===========================================================================================================================
# Main
#
def main():
    args = parse_args()

    if args.precision == "int8":
        calib = CalibrationConfig(
            dataset     = args.calib_path,
            cache       = args.calib_cache,
            image_size  = args.calib_image_size,
            batch_size  = args.calib_batch_size,
            max_batches = args.calib_max_batch
        )
    else:
        calib = None

    build_engine(args.onnx, args.engine, args.precision, calib)

if __name__ == "__main__":
    main()