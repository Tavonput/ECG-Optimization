import argparse

from Quantization.build_engine import build_engine

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX file")
    parser.add_argument("--engine", type=str, default="./engine.engine", help="Path to save engine")
    args = parser.parse_args()

    if not args.onnx:
        print("[ERROR]: ONNX model not provided")
        exit()

    return args

# ---------------------------------------------------------------------------------------------------------------------
# Main
#
def main():
    args = parse_args()
    build_engine(args.onnx, args.engine)

if __name__ == "__main__":
    main()