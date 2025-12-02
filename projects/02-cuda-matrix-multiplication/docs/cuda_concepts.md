# CUDA Matrix Multiplication Concepts

## Memory Hierarchy

### 1. Registers (Fastest)
- **Latency**: 1 cycle
- **Scope**: Thread-private
- **Size**: 255 registers per thread (architecture dependent)
- **Usage**: Local variables, accumulators

### 2. Shared Memory
- **Latency**: ~5 cycles
- **Scope**: Block-shared
- **Size**: 48KB-164KB per SM
- **Usage**: Tile caching, inter-thread communication

### 3. L1/L2 Cache
- **L1 Latency**: ~28 cycles
- **L2 Latency**: ~200 cycles
- **Size**: L1: 128KB per SM, L2: 6MB (A100)
- **Usage**: Automatic caching

### 4. Global Memory
- **Latency**: 200-800 cycles
- **Scope**: Grid-wide
- **Size**: 16GB-80GB (device dependent)
- **Usage**: Main data storage

## Optimization Techniques

### 1. Shared Memory Tiling
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Load tile from global to shared memory
As[ty][tx] = A[row * N + k * TILE_SIZE + tx];
Bs[ty][tx] = B[(k * TILE_SIZE + ty) * N + col];
__syncthreads();

// Compute using shared memory
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}
```

**Benefits**:
- Reduces global memory accesses by factor of TILE_SIZE
- Improves data reuse within thread block

### 2. Memory Coalescing
```cuda
// Good: Consecutive threads access consecutive memory
float val = A[threadIdx.x + blockIdx.x * blockDim.x];

// Bad: Strided access pattern
float val = A[threadIdx.x * stride];
```

**Requirements**:
- Threads in a warp should access contiguous memory
- Alignment to 32, 64, or 128-byte segments

### 3. Register Blocking
```cuda
// Each thread computes multiple outputs
float Csub[WORK_PER_THREAD][WORK_PER_THREAD];

for (int i = 0; i < WORK_PER_THREAD; i++) {
    for (int j = 0; j < WORK_PER_THREAD; j++) {
        Csub[i][j] = 0.0f;
    }
}
```

**Benefits**:
- Increases arithmetic intensity
- Reduces instruction overhead
- Better register utilization

### 4. Bank Conflict Avoidance
```cuda
// Padding to avoid bank conflicts
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];
```

**Explanation**:
- Shared memory has 32 banks
- Successive 32-bit words map to successive banks
- Padding prevents threads from accessing same bank

### 5. Loop Unrolling
```cuda
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}
```

**Benefits**:
- Reduces loop overhead
- Enables instruction-level parallelism
- Compiler optimization

## Performance Metrics

### GFLOPS Calculation
```
GFLOPS = (2 * N^3) / (time_ms * 10^6)
```
- Factor of 2: multiply + add per element
- N^3: Total operations for NxN matrix multiplication

### Memory Bandwidth
```
Bandwidth (GB/s) = bytes_transferred / (time_ms * 10^6)
```

### Arithmetic Intensity
```
AI = FLOPs / bytes_transferred
```
- Naive: ~0.25 ops/byte
- Tiled: ~8 ops/byte
- Optimized: ~16 ops/byte

## GPU Architecture

### Streaming Multiprocessor (SM)
- **Warp Size**: 32 threads
- **Max Threads**: 1024-2048 per SM
- **Max Blocks**: 16-32 per SM
- **Shared Memory**: 48KB-164KB

### Compute Capabilities
| GPU | Compute | SMs | Memory | Peak TFLOPS |
|-----|---------|-----|---------|-------------|
| T4 | 7.5 | 40 | 16GB | 8.1 |
| V100 | 7.0 | 80 | 16GB | 15.7 |
| A100 | 8.0 | 108 | 40GB | 19.5 |
| RTX 3090 | 8.6 | 82 | 24GB | 35.6 |

## Occupancy

### Factors Affecting Occupancy
1. **Registers per thread**
2. **Shared memory per block**
3. **Block size**

### Occupancy Calculation
```
Active Warps = min(
    MaxWarpsPerSM,
    floor(RegistersPerSM / (RegistersPerThread * WarpSize)),
    floor(SharedMemPerSM / SharedMemPerBlock) * WarpsPerBlock
)

Occupancy = Active Warps / MaxWarpsPerSM
```

## Matrix Multiplication Evolution

### 1. Naive Implementation
- **Memory Access**: O(N³) reads
- **Performance**: ~50 GFLOPS
- **Bottleneck**: Memory bandwidth

### 2. Tiled Implementation
- **Memory Access**: O(N³/TILE_SIZE) reads
- **Performance**: ~500 GFLOPS
- **Bottleneck**: Shared memory latency

### 3. Optimized Implementation
- **Memory Access**: Minimized with all techniques
- **Performance**: ~5000 GFLOPS
- **Bottleneck**: Compute throughput

### 4. cuBLAS
- **Memory Access**: Highly optimized
- **Performance**: ~7000 GFLOPS
- **Features**: Tensor cores, mixed precision

## Best Practices

### 1. Memory Access
- Ensure coalesced access patterns
- Minimize global memory transactions
- Use shared memory for data reuse
- Align data structures

### 2. Thread Organization
- Use multiple of warp size (32)
- Balance block size with occupancy
- Consider register pressure

### 3. Synchronization
- Minimize __syncthreads() calls
- Use warp-level primitives when possible
- Avoid divergent branches

### 4. Compilation
- Use appropriate architecture flags (-arch)
- Enable optimizations (-O3)
- Consider -use_fast_math for performance

## Profiling Tools

### Nsight Systems
- Timeline visualization
- API trace
- Memory transfers
- Kernel execution

### Nsight Compute
- Detailed kernel analysis
- Memory throughput
- Instruction throughput
- Occupancy analysis

### Key Metrics to Monitor
1. **SM Efficiency**: % of peak compute
2. **Memory Bandwidth**: % of theoretical peak
3. **Occupancy**: Active warps / max warps
4. **Cache Hit Rate**: L1/L2 effectiveness
5. **Bank Conflicts**: Shared memory efficiency

## Common Pitfalls

### 1. Uncoalesced Memory Access
- **Problem**: Strided or random access
- **Solution**: Restructure data layout

### 2. Bank Conflicts
- **Problem**: Multiple threads access same bank
- **Solution**: Padding or different access pattern

### 3. Low Occupancy
- **Problem**: Too many registers or shared memory
- **Solution**: Reduce resource usage or block size

### 4. Warp Divergence
- **Problem**: Different threads take different paths
- **Solution**: Restructure conditionals

### 5. Register Spilling
- **Problem**: Too many local variables
- **Solution**: Reduce register usage

## Further Reading

1. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
2. [Matrix Multiplication Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
4. [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)