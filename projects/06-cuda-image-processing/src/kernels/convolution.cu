#include "cuda_utils.cuh"
#include <cuda_runtime.h>

#define MAX_KERNEL_SIZE 15
#define MAX_KERNEL_ELEMENTS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)
#define TILE_SIZE 16
#define SHARED_SIZE (TILE_SIZE + MAX_KERNEL_SIZE - 1)

// Constant memory for convolution kernel
__constant__ float c_conv_kernel[MAX_KERNEL_ELEMENTS];

// Method 1: Naive global memory implementation
__global__ void convolution_naive(const unsigned char* input, unsigned char* output,
                                 int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int px = x + kx - half_kernel;
                int py = y + ky - half_kernel;
                
                // Clamp to image boundaries
                px = max(0, min(width - 1, px));
                py = max(0, min(height - 1, py));
                
                float weight = c_conv_kernel[ky * kernel_size + kx];
                sum += weight * input[py * width + px];
            }
        }
        
        // Clamp output to valid range
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum)));
    }
}

// Method 2: Shared memory tiled implementation
__global__ void convolution_shared(const unsigned char* input, unsigned char* output,
                                  int width, int height, int kernel_size) {
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int half_kernel = kernel_size / 2;
    
    // Load tile with halo region
    int load_x_start = blockIdx.x * blockDim.x - half_kernel;
    int load_y_start = blockIdx.y * blockDim.y - half_kernel;
    
    for (int dy = 0; dy < SHARED_SIZE; dy += blockDim.y) {
        for (int dx = 0; dx < SHARED_SIZE; dx += blockDim.x) {
            int shared_x = tx + dx;
            int shared_y = ty + dy;
            
            if (shared_x < SHARED_SIZE && shared_y < SHARED_SIZE) {
                int global_x = load_x_start + shared_x;
                int global_y = load_y_start + shared_y;
                
                // Clamp to boundaries
                global_x = max(0, min(width - 1, global_x));
                global_y = max(0, min(height - 1, global_y));
                
                shared_data[shared_y][shared_x] = input[global_y * width + global_x];
            }
        }
    }
    __syncthreads();
    
    // Compute convolution using shared memory
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int shared_x = tx + half_kernel + kx;
                int shared_y = ty + half_kernel + ky;
                
                float weight = c_conv_kernel[ky * kernel_size + kx];
                sum += weight * shared_data[shared_y][shared_x];
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum)));
    }
}

// Method 3: Optimized shared memory with boundary handling
__global__ void convolution_optimized(const unsigned char* input, unsigned char* output,
                                     int width, int height, int kernel_size, int boundary_mode) {
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int half_kernel = kernel_size / 2;
    
    // Load tile with halo region
    int load_x_start = blockIdx.x * blockDim.x - half_kernel;
    int load_y_start = blockIdx.y * blockDim.y - half_kernel;
    
    for (int dy = 0; dy < SHARED_SIZE; dy += blockDim.y) {
        for (int dx = 0; dx < SHARED_SIZE; dx += blockDim.x) {
            int shared_x = tx + dx;
            int shared_y = ty + dy;
            
            if (shared_x < SHARED_SIZE && shared_y < SHARED_SIZE) {
                int global_x = load_x_start + shared_x;
                int global_y = load_y_start + shared_y;
                
                float value = 0.0f;
                
                // Handle different boundary conditions
                if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                    value = input[global_y * width + global_x];
                } else {
                    switch (boundary_mode) {
                        case 0: // Zero padding
                            value = 0.0f;
                            break;
                        case 1: // Clamp/extend
                            global_x = max(0, min(width - 1, global_x));
                            global_y = max(0, min(height - 1, global_y));
                            value = input[global_y * width + global_x];
                            break;
                        case 2: // Reflect
                            if (global_x < 0) global_x = -global_x;
                            if (global_x >= width) global_x = 2 * width - global_x - 2;
                            if (global_y < 0) global_y = -global_y;
                            if (global_y >= height) global_y = 2 * height - global_y - 2;
                            global_x = max(0, min(width - 1, global_x));
                            global_y = max(0, min(height - 1, global_y));
                            value = input[global_y * width + global_x];
                            break;
                        case 3: // Replicate
                            global_x = max(0, min(width - 1, global_x));
                            global_y = max(0, min(height - 1, global_y));
                            value = input[global_y * width + global_x];
                            break;
                    }
                }
                
                shared_data[shared_y][shared_x] = value;
            }
        }
    }
    __syncthreads();
    
    // Compute convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int shared_x = tx + half_kernel + kx;
                int shared_y = ty + half_kernel + ky;
                
                float weight = c_conv_kernel[ky * kernel_size + kx];
                sum += weight * shared_data[shared_y][shared_x];
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum)));
    }
}

// Method 4: RGB convolution (process all channels)
__global__ void convolution_rgb(const unsigned char* input, unsigned char* output,
                               int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int px = x + kx - half_kernel;
                int py = y + ky - half_kernel;
                
                // Clamp to boundaries
                px = max(0, min(width - 1, px));
                py = max(0, min(height - 1, py));
                
                int idx = (py * width + px) * 3;
                float weight = c_conv_kernel[ky * kernel_size + kx];
                
                sum_r += weight * input[idx];
                sum_g += weight * input[idx + 1];
                sum_b += weight * input[idx + 2];
            }
        }
        
        int out_idx = (y * width + x) * 3;
        output[out_idx] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum_r)));
        output[out_idx + 1] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum_g)));
        output[out_idx + 2] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, sum_b)));
    }
}

// Host utility functions
void normalize_kernel(float* kernel, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size * size; i++) {
        sum += kernel[i];
    }
    if (sum != 0.0f) {
        for (int i = 0; i < size * size; i++) {
            kernel[i] /= sum;
        }
    }
}

void create_box_filter(float* kernel, int size) {
    float value = 1.0f / (size * size);
    for (int i = 0; i < size * size; i++) {
        kernel[i] = value;
    }
}

void create_sharpen_filter(float* kernel) {
    // 3x3 sharpen kernel
    kernel[0] = 0.0f; kernel[1] = -1.0f; kernel[2] = 0.0f;
    kernel[3] = -1.0f; kernel[4] = 5.0f; kernel[5] = -1.0f;
    kernel[6] = 0.0f; kernel[7] = -1.0f; kernel[8] = 0.0f;
}

void create_edge_detection_filter(float* kernel) {
    // 3x3 edge detection kernel
    kernel[0] = -1.0f; kernel[1] = -1.0f; kernel[2] = -1.0f;
    kernel[3] = -1.0f; kernel[4] = 8.0f; kernel[5] = -1.0f;
    kernel[6] = -1.0f; kernel[7] = -1.0f; kernel[8] = -1.0f;
}

void create_emboss_filter(float* kernel) {
    // 3x3 emboss kernel
    kernel[0] = -2.0f; kernel[1] = -1.0f; kernel[2] = 0.0f;
    kernel[3] = -1.0f; kernel[4] = 1.0f; kernel[5] = 1.0f;
    kernel[6] = 0.0f; kernel[7] = 1.0f; kernel[8] = 2.0f;
}

// Host wrapper functions
extern "C" {

void setup_convolution_kernel(const float* kernel, int kernel_size) {
    if (kernel_size > MAX_KERNEL_SIZE) {
        kernel_size = MAX_KERNEL_SIZE;
    }
    
    int kernel_elements = kernel_size * kernel_size;
    CUDA_CHECK(cudaMemcpyToSymbol(c_conv_kernel, kernel, 
                                  kernel_elements * sizeof(float)));
}

void launch_convolution_naive(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, int kernel_size,
                             cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    convolution_naive<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                            width, height, kernel_size);
    CUDA_CHECK_KERNEL();
}

void launch_convolution_shared(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, int kernel_size,
                              cudaStream_t stream = 0) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    convolution_shared<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                             width, height, kernel_size);
    CUDA_CHECK_KERNEL();
}

void launch_convolution_optimized(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, int kernel_size, int boundary_mode,
                                 cudaStream_t stream = 0) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    convolution_optimized<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                               width, height, kernel_size, 
                                                               boundary_mode);
    CUDA_CHECK_KERNEL();
}

void launch_convolution_rgb(const unsigned char* d_input, unsigned char* d_output,
                           int width, int height, int kernel_size,
                           cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    convolution_rgb<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                          width, height, kernel_size);
    CUDA_CHECK_KERNEL();
}

// Utility function to choose best implementation
void launch_convolution_auto(const unsigned char* d_input, unsigned char* d_output,
                            int width, int height, int kernel_size, int boundary_mode,
                            cudaStream_t stream = 0) {
    if (kernel_size > MAX_KERNEL_SIZE) {
        kernel_size = MAX_KERNEL_SIZE;
    }
    
    int total_pixels = width * height;
    
    if (kernel_size >= 7 && total_pixels > 1920 * 1080) {
        // Large kernel and image: use optimized version
        launch_convolution_optimized(d_input, d_output, width, height, 
                                   kernel_size, boundary_mode, stream);
    } else if (total_pixels > 720 * 480) {
        // Medium image: use shared memory
        launch_convolution_shared(d_input, d_output, width, height, kernel_size, stream);
    } else {
        // Small image: naive implementation
        launch_convolution_naive(d_input, d_output, width, height, kernel_size, stream);
    }
}

// Convenience functions for common filters
void launch_box_filter(const unsigned char* d_input, unsigned char* d_output,
                      int width, int height, int filter_size, cudaStream_t stream = 0) {
    float* h_kernel = new float[filter_size * filter_size];
    create_box_filter(h_kernel, filter_size);
    setup_convolution_kernel(h_kernel, filter_size);
    launch_convolution_auto(d_input, d_output, width, height, filter_size, 1, stream);
    delete[] h_kernel;
}

void launch_sharpen_filter(const unsigned char* d_input, unsigned char* d_output,
                          int width, int height, cudaStream_t stream = 0) {
    float h_kernel[9];
    create_sharpen_filter(h_kernel);
    setup_convolution_kernel(h_kernel, 3);
    launch_convolution_auto(d_input, d_output, width, height, 3, 1, stream);
}

void launch_edge_detection_filter(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, cudaStream_t stream = 0) {
    float h_kernel[9];
    create_edge_detection_filter(h_kernel);
    setup_convolution_kernel(h_kernel, 3);
    launch_convolution_auto(d_input, d_output, width, height, 3, 1, stream);
}

void launch_emboss_filter(const unsigned char* d_input, unsigned char* d_output,
                         int width, int height, cudaStream_t stream = 0) {
    float h_kernel[9];
    create_emboss_filter(h_kernel);
    setup_convolution_kernel(h_kernel, 3);
    launch_convolution_auto(d_input, d_output, width, height, 3, 1, stream);
}

} // extern "C"