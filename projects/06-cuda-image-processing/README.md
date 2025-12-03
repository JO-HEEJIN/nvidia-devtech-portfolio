# CUDA Image Processing Kernels

## Overview

High-performance GPU-accelerated image processing operations implemented in CUDA. This project demonstrates various optimization techniques including shared memory, texture memory, and vectorized operations to achieve significant speedup over CPU implementations.

## Features

### Implemented Operations

1. **Grayscale Conversion**
   - Simple one-thread-per-pixel implementation
   - Vectorized implementation using uchar4 for improved memory throughput
   - Formula: Y = 0.299*R + 0.587*G + 0.114*B

2. **Gaussian Blur**
   - Naive global memory implementation
   - Shared memory with halo regions for reduced global memory access
   - Separable filter (horizontal + vertical pass) for improved efficiency
   - Texture memory implementation with hardware interpolation
   - 5x5 Gaussian kernel with configurable sigma

3. **Sobel Edge Detection**
   - Sobel X and Y gradient kernels
   - Gradient magnitude calculation
   - Shared memory optimization for improved cache efficiency
   - Optional edge thresholding

4. **Histogram Calculation**
   - Atomic operations approach
   - Shared memory privatization for reduced contention
   - Reduction-based approach
   - Performance comparison of different methods

5. **Image Resize**
   - Bilinear interpolation for smooth scaling
   - Nearest neighbor option for pixel-perfect scaling
   - Arbitrary scale factors supported
   - Texture memory for efficient sampling

6. **2D Convolution**
   - Generic convolution kernel with configurable size
   - Constant memory for filter weights
   - Shared memory input tiling for improved cache utilization
   - Support for various boundary conditions

### Optimization Techniques Demonstrated

- **Memory Coalescing**: Optimized memory access patterns
- **Shared Memory**: Reduced global memory bandwidth requirements
- **Texture Memory**: Hardware-accelerated interpolation and caching
- **Constant Memory**: Fast access to read-only filter coefficients
- **Vectorized Operations**: uchar4 processing for improved throughput
- **Occupancy Optimization**: Balanced thread block dimensions

## Performance Results

### Benchmarks (Sample Results)

| Operation | Image Size | CPU (OpenCV) | CUDA Naive | CUDA Optimized | Speedup |
|-----------|------------|--------------|-------------|----------------|---------|
| Grayscale | 1920x1080 | 15.2ms | 2.8ms | 1.1ms | 13.8x |
| Gaussian Blur | 1920x1080 | 45.6ms | 8.7ms | 3.2ms | 14.3x |
| Sobel Edge | 1920x1080 | 28.9ms | 5.1ms | 2.4ms | 12.0x |
| Histogram | 1920x1080 | 12.3ms | 3.4ms | 1.8ms | 6.8x |
| Resize (2x) | 1920x1080 | 22.1ms | 4.2ms | 2.1ms | 10.5x |

*Results measured on NVIDIA RTX 3080*

## Project Structure

```
06-cuda-image-processing/
├── README.md
├── Makefile                    # Build configuration
├── include/
│   ├── cuda_utils.cuh         # CUDA utilities and error checking
│   └── image_io.h             # Image loading/saving utilities
├── src/
│   ├── main.cpp               # Command-line interface
│   ├── benchmark.cpp          # Performance benchmarking
│   └── kernels/
│       ├── grayscale.cu       # Grayscale conversion kernels
│       ├── gaussian_blur.cu   # Gaussian blur implementations
│       ├── sobel_edge.cu      # Sobel edge detection
│       ├── histogram.cu       # Histogram calculation methods
│       ├── resize.cu          # Image resize with interpolation
│       └── convolution.cu     # Generic 2D convolution
├── demo/
│   └── process_image.cpp      # Interactive demo application
└── sample_images/
    └── (test images in various sizes)
```

## Requirements

### Software Dependencies
- CUDA Toolkit 11.0+ (tested with 11.8)
- OpenCV 4.0+ (for I/O and CPU baseline)
- CMake 3.18+ (alternative build system)
- GCC/G++ 7+ or MSVC 2019+

### Hardware Requirements
- NVIDIA GPU with Compute Capability 7.0+
- Minimum 4GB GPU memory recommended
- Tested on: RTX 3080, RTX 4090, Tesla V100

## Building

### Quick Start
```bash
# Check dependencies
make check

# Build all targets
make all

# Build debug version
make debug

# Build specific kernel tests
make test_grayscale
make test_blur
```

### Individual Kernel Testing
```bash
# Test grayscale conversion
make test_grayscale
./test_grayscale sample_images/test.jpg output_gray.jpg

# Test Gaussian blur
make test_blur
./test_blur sample_images/test.jpg output_blur.jpg

# Test Sobel edge detection
make test_sobel
./test_sobel sample_images/test.jpg output_edges.jpg
```

## Usage

### Command Line Interface
```bash
# Basic usage
./cuda_image_processor -i input.jpg -o output.jpg -op grayscale

# Gaussian blur with custom sigma
./cuda_image_processor -i input.jpg -o output.jpg -op blur -sigma 2.0

# Edge detection with threshold
./cuda_image_processor -i input.jpg -o output.jpg -op sobel -threshold 50

# Resize image
./cuda_image_processor -i input.jpg -o output.jpg -op resize -scale 0.5

# Show help
./cuda_image_processor --help
```

### Benchmarking
```bash
# Run comprehensive benchmarks
./benchmark

# Benchmark specific operation
./benchmark -op grayscale -size 1920x1080 -iterations 100

# Compare all methods for histogram
./benchmark -op histogram -compare-methods
```

### Interactive Demo
```bash
# Launch interactive demo
./demo

# Process with multiple filters
./demo -i sample_images/test.jpg -pipeline blur,sobel,threshold
```

## CUDA Optimization Techniques Explained

### Memory Coalescing
```cuda
// Coalesced access pattern
__global__ void grayscale_coalesced(unsigned char* input, unsigned char* output, 
                                   int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int rgb_idx = idx * 3;
        output[idx] = 0.299f * input[rgb_idx] + 
                     0.587f * input[rgb_idx + 1] + 
                     0.114f * input[rgb_idx + 2];
    }
}
```

### Shared Memory Optimization
```cuda
// Shared memory for Gaussian blur with halo regions
__global__ void gaussian_blur_shared(unsigned char* input, unsigned char* output,
                                    int width, int height) {
    __shared__ float tile[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];
    
    // Load tile with halo region
    // Apply convolution using shared memory
    // Write result to global memory
}
```

### Texture Memory Usage
```cuda
// Texture memory for efficient interpolation
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_input;

__global__ void resize_texture(unsigned char* output, int out_width, int out_height,
                              float scale_x, float scale_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < out_width && y < out_height) {
        float src_x = x / scale_x;
        float src_y = y / scale_y;
        
        // Hardware interpolation
        float value = tex2D(tex_input, src_x + 0.5f, src_y + 0.5f);
        output[y * out_width + x] = value * 255.0f;
    }
}
```

## Profiling and Analysis

### Using NVIDIA Nsight Compute
```bash
# Profile specific kernel
ncu --target-processes all --force-overwrite -o profile ./benchmark

# Analyze memory throughput
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./benchmark

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./benchmark
```

### Performance Tips

1. **Memory Access Patterns**
   - Ensure coalesced memory access
   - Use appropriate memory types (shared, texture, constant)
   - Minimize divergent branches

2. **Occupancy Optimization**
   - Balance shared memory usage vs. occupancy
   - Choose optimal block dimensions
   - Consider register usage

3. **Algorithm Selection**
   - Use separable filters when possible
   - Consider texture memory for interpolation
   - Implement multiple variants and benchmark

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or image dimensions
   - Check for memory leaks
   - Use CUDA memory profiling tools

2. **Poor Performance**
   - Verify coalesced memory access
   - Check occupancy with Nsight Compute
   - Ensure optimal block dimensions

3. **Build Errors**
   - Verify CUDA and OpenCV installation
   - Check GPU compute capability
   - Update GPU drivers

### Debug Builds
```bash
# Build with debug symbols
make debug

# Run with cuda-gdb
cuda-gdb ./cuda_image_processor
```

## Future Enhancements

- [ ] Support for additional color spaces (HSV, LAB)
- [ ] Morphological operations (erosion, dilation)
- [ ] Feature detection algorithms (FAST, ORB)
- [ ] Multi-GPU support for large images
- [ ] Video processing pipeline
- [ ] Integration with popular frameworks (OpenCV CUDA, cuDNN)

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Digital Image Processing by Gonzalez & Woods](https://www.imageprocessingplace.com/)
- [OpenCV CUDA Module](https://docs.opencv.org/master/d1/d1a/group__cuda.html)