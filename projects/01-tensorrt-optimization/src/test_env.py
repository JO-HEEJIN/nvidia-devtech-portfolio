import sys
import numpy as np
import torch
import torchvision

print(f"Python version: {sys.version}")
print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

try:
    import tensorrt
    print(f"TensorRT version: {tensorrt.__version__}")
except ImportError:
    print("TensorRT not found (expected on macOS)")

print("\nEnvironment check passed!")
