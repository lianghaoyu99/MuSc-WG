import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch built with CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("\nReason for no CUDA:")
    if torch.version.cuda is None:
        print("-> PyTorch was installed without CUDA support (CPU-only version).")
    else:
        print("-> PyTorch has CUDA support, but cannot find a GPU.")
        print("   Possible causes: Old drivers, wrong CUDA version match, or hardware issues.")
