# CUDA Interview Preparation

## Core Concepts

### Memory Hierarchy
- Global memory
- Shared memory
- Constant memory
- Texture memory
- Registers

### Thread Organization
- Threads, blocks, and grids
- Warp execution
- Thread divergence
- Occupancy optimization

### Common Patterns
- Reduction
- Scan (prefix sum)
- Matrix multiplication
- Convolution

---

## Practice Problems

### Basic
- [ ] Vector addition
- [ ] Matrix transpose
- [ ] Element-wise operations

### Intermediate
- [ ] Matrix multiplication (tiled)
- [ ] Reduction operations
- [ ] Histogram calculation

### Advanced
- [ ] Dynamic parallelism
- [ ] Streams and concurrency
- [ ] Multi-GPU programming

---

## Key Questions

1. Explain the CUDA memory hierarchy
2. What is warp divergence and how to avoid it?
3. How does shared memory improve performance?
4. Explain coalesced memory access
5. What is occupancy and how to optimize it?
6. Difference between __syncthreads() and __threadfence()?
7. How to profile CUDA code?
8. Explain atomic operations and their use cases

---

## Resources

- CUDA Programming Guide
- CUDA Best Practices Guide
- NVIDIA Nsight Compute
- CUDA Samples
