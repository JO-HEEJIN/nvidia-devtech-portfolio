# AI Optimization Interview Preparation

## TensorRT

### Key Concepts
- Layer fusion
- Precision calibration (FP32, FP16, INT8)
- Dynamic shapes
- Plugins and custom layers
- Builder optimization profiles

### Common Questions
1. How does TensorRT optimize models?
2. Explain INT8 calibration process
3. What is layer fusion?
4. How to handle dynamic batch sizes?
5. When to use custom plugins?

---

## Quantization

### Techniques
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Per-channel vs per-tensor quantization
- Symmetric vs asymmetric quantization

### Common Questions
1. Difference between PTQ and QAT?
2. How to minimize accuracy loss?
3. Explain calibration dataset selection
4. What layers are sensitive to quantization?

---

## Model Deployment

### Triton Inference Server
- Model repository structure
- Dynamic batching
- Model ensembles
- Backend selection

### Common Questions
1. Why use Triton over custom serving?
2. How does dynamic batching work?
3. Explain model versioning
4. How to monitor inference performance?

---

## Performance Optimization

### Metrics
- Latency
- Throughput
- GPU utilization
- Memory bandwidth

### Techniques
- Batching strategies
- Mixed precision inference
- Kernel fusion
- Memory optimization

---

## Resources

- TensorRT Developer Guide
- Triton Inference Server Documentation
- NVIDIA Deep Learning Performance Guide
