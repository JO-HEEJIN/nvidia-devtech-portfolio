# INT8 Quantization Pipeline

A comprehensive implementation demonstrating neural network quantization techniques using PyTorch and TensorRT.

## Overview

This project implements both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) methods to reduce model size and improve inference speed while maintaining accuracy.

## Quantization Theory

### Neural Network Quantization

Quantization reduces the precision of weights and activations from 32-bit floating point (FP32) to lower bit representations like 8-bit integers (INT8). This provides several benefits:

- **Model Size Reduction**: 3-4x smaller models
- **Inference Speedup**: 2-4x faster inference on hardware with INT8 support
- **Memory Bandwidth**: Reduced memory access requirements
- **Power Efficiency**: Lower energy consumption

### Quantization Formula

The quantization process maps floating point values to quantized integers:

```
quantized = round(scale * (float_val - zero_point))
float_val = scale * (quantized - zero_point)
```

Where:
- `scale`: Scaling factor to map float range to quantized range
- `zero_point`: Offset to handle asymmetric ranges

## PTQ vs QAT Comparison

| Aspect | Post-Training Quantization (PTQ) | Quantization-Aware Training (QAT) |
|--------|----------------------------------|-----------------------------------|
| **Training Required** | No | Yes |
| **Accuracy** | Good (1-3% drop) | Better (0.5-1% drop) |
| **Time to Deploy** | Fast (minutes) | Slow (hours/days) |
| **Computational Cost** | Low | High |
| **Use Case** | Quick deployment | Maximum accuracy |
| **Calibration Data** | Small subset needed | Full training data |

### Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained FP32 model without retraining:

1. **Calibration**: Run representative data through the model
2. **Statistics Collection**: Gather activation ranges for each layer
3. **Scale Computation**: Calculate optimal quantization parameters
4. **Model Conversion**: Apply quantization to weights and activations

### Quantization-Aware Training (QAT)

QAT simulates quantization during training:

1. **Fake Quantization**: Apply quantization/dequantization during forward pass
2. **Gradient Flow**: Maintain FP32 gradients for backward pass
3. **Parameter Updates**: Update FP32 weights normally
4. **Model Conversion**: Convert to true quantized model after training

## Calibration Methods

### Entropy Calibration (KL-Divergence)

Minimizes information loss by finding the threshold that preserves the statistical distribution:

```python
# TensorRT IInt8EntropyCalibrator2
# Minimizes KL-divergence between original and quantized distributions
threshold = argmin(KL_divergence(original_dist, quantized_dist))
```

**Pros**: Better accuracy preservation, optimal for most layers
**Cons**: Slower calibration process, requires more computation

### MinMax Calibration

Uses the absolute maximum value as the quantization range:

```python
# Simple range mapping
threshold = max(abs(activations))
scale = threshold / 127  # For INT8 range [-128, 127]
```

**Pros**: Fast calibration, simple implementation
**Cons**: Sensitive to outliers, may waste quantization range

### Percentile Calibration

Uses a percentile (e.g., 99.9%) to handle outliers:

```python
threshold = np.percentile(abs(activations), 99.9)
```

**Pros**: Robust to outliers, good balance of speed and accuracy
**Cons**: May lose some precision for extreme values

## Project Structure

```
05-int8-quantization/
├── README.md
├── requirements.txt
├── src/
│   ├── calibration_dataset.py    # ImageNet calibration data
│   ├── ptq_tensorrt.py          # TensorRT post-training quantization
│   ├── qat_pytorch.py           # PyTorch quantization-aware training
│   ├── sensitivity_analysis.py  # Per-layer sensitivity analysis
│   ├── mixed_precision.py       # Optimal precision assignment
│   ├── accuracy_evaluation.py   # Comprehensive accuracy metrics
│   └── compare_methods.py       # Method comparison utilities
├── analysis/
│   └── generate_report.py       # Automated report generation
└── notebooks/
    └── quantization_tutorial.ipynb  # Interactive tutorial
```

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Prepare Calibration Data**
```python
from src.calibration_dataset import create_calibration_dataset
calib_loader = create_calibration_dataset(data_path="path/to/imagenet")
```

3. **Run Post-Training Quantization**
```python
from src.ptq_tensorrt import quantize_model_ptq
quantized_model = quantize_model_ptq(model, calib_loader)
```

4. **Run Quantization-Aware Training**
```python
from src.qat_pytorch import train_quantized_model
qat_model = train_quantized_model(model, train_loader, val_loader)
```

## Supported Models

- **ResNet50**: Standard CNN architecture
- **VGG16**: Deep convolutional network
- **EfficientNet-B0**: Efficient mobile architecture

## Target Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Accuracy Drop | < 1% | TBD |
| Model Size Reduction | 3-4x | TBD |
| Inference Speedup | 2-4x | TBD |
| Memory Usage | 70-75% reduction | TBD |

## Best Practices

1. **Calibration Data**: Use diverse, representative samples (1000+ images)
2. **Layer Selection**: Keep sensitive layers (first/last) in higher precision
3. **Validation**: Always validate on full test set, not just calibration data
4. **Hardware**: Test on target deployment hardware for realistic performance
5. **Monitoring**: Track both accuracy and inference metrics

## Common Issues and Solutions

### Accuracy Degradation
- Increase calibration dataset size
- Use mixed precision for sensitive layers
- Try different calibration methods
- Consider quantization-aware training

### Poor Performance
- Verify INT8 hardware support
- Check batch size optimization
- Profile memory bandwidth utilization
- Validate kernel implementations

## References

- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [NVIDIA Model Optimizer](https://docs.nvidia.com/deeplearning/tensorrt/model-optimizer/)
- [Quantization Papers and Research](https://arxiv.org/abs/1712.05877)