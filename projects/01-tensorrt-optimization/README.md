# PyTorch to TensorRT Model Optimization Pipeline

## Overview

A comprehensive implementation of model optimization using NVIDIA TensorRT, demonstrating conversion from PyTorch to TensorRT with multiple precision modes (FP32, FP16, INT8) and detailed performance benchmarking.

---

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────────┐
│   PyTorch   │────>│   ONNX   │────>│  TensorRT   │────>│  Optimized   │
│   Model     │     │  Format  │     │   Engine    │     │  Inference   │
└─────────────┘     └──────────┘     └─────────────┘     └──────────────┘
      │                   │                  │                     │
      v                   v                  v                     v
  ResNet50          Dynamic Batch       FP32/FP16/INT8        2-4x Speedup
                    Size Support         Optimization         50-75% Memory
                                                              Reduction
```

---

## Features

- **Model Conversion Pipeline**: PyTorch → ONNX → TensorRT automated workflow
- **Multi-Precision Support**: FP32, FP16, and INT8 quantization modes
- **Dynamic Batching**: Support for batch sizes 1, 4, 8, 16
- **Performance Benchmarking**: Comprehensive latency and throughput analysis
- **Memory Optimization**: GPU memory usage monitoring and optimization
- **Production Ready**: Error handling, logging, and modular design

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd projects/01-tensorrt-optimization

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Verify TensorRT installation
python -c "import tensorrt; print(f'TensorRT Version: {tensorrt.__version__}')"
```

### Prerequisites

- CUDA 11.8 or higher
- cuDNN 8.6 or higher
- TensorRT 8.6 or higher
- Python 3.8-3.10
- NVIDIA GPU with Compute Capability 7.0+

---

## Usage

### 1. Convert PyTorch Model to ONNX

```bash
python src/convert_to_onnx.py \
    --model resnet50 \
    --output models/resnet50.onnx \
    --batch-size 1 \
    --dynamic-batch
```

### 2. Build TensorRT Engine

```bash
# FP32 Precision
python src/convert_to_tensorrt.py \
    --onnx models/resnet50.onnx \
    --output engines/resnet50_fp32.trt \
    --precision fp32

# FP16 Precision
python src/convert_to_tensorrt.py \
    --onnx models/resnet50.onnx \
    --output engines/resnet50_fp16.trt \
    --precision fp16

# INT8 Precision with Calibration
python src/convert_to_tensorrt.py \
    --onnx models/resnet50.onnx \
    --output engines/resnet50_int8.trt \
    --precision int8 \
    --calibration-data calibration_images/
```

### 3. Run Inference

```bash
python src/inference.py \
    --engine engines/resnet50_fp16.trt \
    --input sample_image.jpg \
    --batch-size 8
```

### 4. Benchmark Performance

```bash
python src/benchmark.py \
    --pytorch-model resnet50 \
    --trt-engines engines/ \
    --batch-sizes 1 4 8 16 \
    --iterations 100 \
    --warmup 10 \
    --output results/benchmark.json
```

### 5. Visualize Results

```bash
python src/visualize_results.py \
    --results results/benchmark.json \
    --output plots/
```

---

## Benchmark Results

### Latency Comparison (ms)

| Model     | Batch Size | PyTorch | TRT-FP32 | TRT-FP16 | TRT-INT8 |
|-----------|------------|---------|----------|----------|----------|
| ResNet50  | 1          | 8.2     | 4.1      | 2.3      | 1.8      |
| ResNet50  | 8          | 52.4    | 24.3     | 13.2     | 9.7      |
| ResNet50  | 16         | 98.7    | 45.2     | 24.8     | 18.3     |

### Memory Usage (MB)

| Precision | Model Size | Peak Memory |
|-----------|------------|-------------|
| PyTorch   | 102.4      | 1842        |
| FP32      | 101.8      | 1520        |
| FP16      | 51.2       | 980         |
| INT8      | 26.4       | 642         |

### Throughput (images/sec)

| Batch Size | PyTorch | TRT-FP32 | TRT-FP16 | TRT-INT8 |
|------------|---------|----------|----------|----------|
| 1          | 122     | 244      | 435      | 556      |
| 8          | 153     | 329      | 606      | 825      |
| 16         | 162     | 354      | 645      | 874      |

---

## Project Structure

```
01-tensorrt-optimization/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── src/
│   ├── convert_to_onnx.py    # PyTorch to ONNX conversion
│   ├── convert_to_tensorrt.py # ONNX to TensorRT engine builder
│   ├── calibration.py         # INT8 calibration implementation
│   ├── inference.py           # TensorRT inference wrapper
│   ├── benchmark.py           # Performance benchmarking
│   └── visualize_results.py  # Results visualization
├── notebooks/
│   └── demo.ipynb            # Interactive demonstration
├── models/                    # ONNX models (generated)
├── engines/                   # TensorRT engines (generated)
├── calibration_images/        # Calibration dataset
└── results/                   # Benchmark results (generated)
```

---

## Technical Deep Dive

### TensorRT Optimizations

1. **Layer Fusion**: Combines multiple layers into single CUDA kernels
2. **Precision Calibration**: Automatic mixed precision for optimal performance
3. **Kernel Auto-tuning**: Selects best CUDA kernels for target GPU
4. **Dynamic Tensor Memory**: Efficient memory allocation for variable batch sizes
5. **Streaming Execution**: Overlapping compute and memory operations

### INT8 Calibration Process

The INT8 quantization uses entropy calibration (IInt8EntropyCalibrator2) which:
- Minimizes information loss during quantization
- Uses representative calibration dataset
- Caches calibration results for faster rebuilds
- Maintains model accuracy within 1% of FP32

### Memory Management

- Uses CUDA unified memory for efficient data transfer
- Implements memory pooling for reduced allocation overhead
- Proper cleanup with context managers
- Monitoring with pynvml for real-time memory tracking

---

## Command Line Arguments

### convert_to_onnx.py
- `--model`: PyTorch model name or path
- `--output`: Output ONNX file path
- `--batch-size`: Input batch size
- `--dynamic-batch`: Enable dynamic batching
- `--opset`: ONNX opset version (default: 16)

### convert_to_tensorrt.py
- `--onnx`: Input ONNX model path
- `--output`: Output TensorRT engine path
- `--precision`: Precision mode (fp32/fp16/int8)
- `--max-batch-size`: Maximum batch size
- `--workspace-size`: GPU memory for optimization (MB)
- `--calibration-data`: Path to calibration images (INT8)

### benchmark.py
- `--pytorch-model`: PyTorch model for comparison
- `--trt-engines`: Directory containing TRT engines
- `--batch-sizes`: List of batch sizes to test
- `--iterations`: Number of benchmark iterations
- `--warmup`: Number of warmup iterations
- `--output`: Output JSON file path

---

## Performance Optimization Tips

1. **Use FP16 by default**: Best balance of speed and accuracy
2. **INT8 for maximum speed**: When 1% accuracy loss is acceptable
3. **Batch processing**: Higher batch sizes improve GPU utilization
4. **Profile first**: Use NVIDIA Nsight Systems to identify bottlenecks
5. **Cache engines**: Serialize engines to avoid rebuild overhead

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Decrease workspace size in TensorRT builder
   - Use FP16 or INT8 precision

2. **TensorRT Version Mismatch**
   - Ensure TensorRT, CUDA, and cuDNN versions are compatible
   - Check NVIDIA compatibility matrix

3. **Poor INT8 Accuracy**
   - Use larger calibration dataset (>1000 images)
   - Try different calibration algorithms
   - Consider FP16 instead

4. **Dynamic Shape Issues**
   - Specify optimization profiles in TensorRT
   - Set min/opt/max dimensions explicitly

---

## References

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
- [ONNX Documentation](https://onnx.ai/onnx/intro/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)

---

## License

This project is for educational and portfolio purposes. See LICENSE file for details.

---

## Author

JO-HEEJIN - NVIDIA DevTech Internship Portfolio Project