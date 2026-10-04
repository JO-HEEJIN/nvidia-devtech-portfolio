# TensorRT Optimization Demo - Conclusions

## Key Findings

### 1. TensorRT Optimization Results
TensorRT engines built and ran successfully at both FP32 and FP16.
Precision mode decides whether that's a win:
- **FP32**: 0.88x vs. PyTorch FP32 — a slowdown. Converting to TensorRT
  with no precision change adds engine overhead without giving the GPU
  anything new to exploit.
- **FP16**: 2.24x vs. PyTorch FP32 (batch 1) — a real speedup, from
  using the GPU's Tensor Cores instead of just running the same ops in
  a different runtime.

### 2. Performance Summary (batch size 1, resnet18, 224x224)
- PyTorch FP32 baseline: 3.95 ms
- TensorRT FP32: 4.50 ms (0.88x — slower than baseline)
- TensorRT FP16: 1.76 ms (2.24x faster than baseline)

The FP16 speedup holds and grows at larger batch sizes:

| Batch | PyTorch FP32 | TensorRT FP16 | Speedup |
|---|---|---|---|
| 1 | 3.95 ms | 1.76 ms | 2.24x |
| 2 | 5.13 ms | 1.98 ms | 2.60x |
| 4 | 8.69 ms | 2.68 ms | 3.24x |

(Full numbers: `demo_benchmark.json`, `summary.txt`.)

### 3. Environment
- GPU: Tesla T4
- TensorRT version: 10.11.0.33 (printed directly via `trt.__version__`
  during the run, cell 10 of the notebook)
- Both FP32 and FP16 engines built and ran without errors

## Recommendations

1. **Use FP16, not FP32-only conversion**: the data above is a direct
   demonstration — FP32-only TensorRT was a net loss here, FP16 was a
   2.2–3.3x win depending on batch size. The optimization comes from
   the precision change hitting Tensor Cores, not from TensorRT itself.
2. **Consider INT8 for even more speedup**: requires calibration data.
3. **Cache engines**: save `.plan` files to avoid rebuild overhead.

---

*Note: an earlier version of this file reported "Tesla P100-PCIE-16GB"
and "TensorRT version: 10.14.1" with a 0.88x headline figure. Both the
GPU and TensorRT version were wrong — this run was on a T4 at TensorRT
10.11.0.33, both confirmed from the notebook's own execution output
(`results/pytorch-to-tensorrt-optimization.ipynb`, cell 10) and
`summary.txt`, which had the correct GPU all along. The 0.88x number
itself wasn't fabricated — it's the real FP32-vs-FP32 result — but the
conclusion-generating code checked FP32 before FP16 and stopped at the
first engine it found, so it never looked at the FP16 result sitting
in the same benchmark run.*
