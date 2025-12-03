#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16
#define SHARED_SIZE (TILE_SIZE + 2)

// Sobel operators in constant memory
__constant__ float c_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float c_sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// Scharr operators (more accurate edge detection)
__constant__ float c_scharr_x[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
__constant__ float c_scharr_y[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};

// Method 1: Naive Sobel edge detection
__global__ void sobel_edge_naive(const unsigned char* input, unsigned char* output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int px = x + kx - 1;
                int py = y + ky - 1;
                int idx = py * width + px;
                
                float pixel = static_cast<float>(input[idx]);
                gx += c_sobel_x[ky * 3 + kx] * pixel;
                gy += c_sobel_y[ky * 3 + kx] * pixel;
            }
        }
        
        // Compute gradient magnitude
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude));
    } else if (x < width && y < height) {
        // Set border pixels to 0
        output[y * width + x] = 0;
    }
}

// Method 2: Shared memory optimization
__global__ void sobel_edge_shared(const unsigned char* input, unsigned char* output,
                                 int width, int height) {
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Load tile with border
    int load_x = x - 1;
    int load_y = y - 1;
    
    // Load main tile
    if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
        shared_data[ty][tx] = input[load_y * width + load_x];
    } else {
        shared_data[ty][tx] = 0.0f;
    }
    
    // Load right halo
    if (tx == blockDim.x - 1) {
        int halo_x = load_x + 1;
        if (halo_x < width && load_y >= 0 && load_y < height) {
            shared_data[ty][tx + 1] = input[load_y * width + halo_x];
        } else {
            shared_data[ty][tx + 1] = 0.0f;
        }
    }
    
    // Load bottom halo
    if (ty == blockDim.y - 1) {
        int halo_y = load_y + 1;
        if (load_x >= 0 && load_x < width && halo_y < height) {
            shared_data[ty + 1][tx] = input[halo_y * width + load_x];
        } else {
            shared_data[ty + 1][tx] = 0.0f;
        }
    }
    
    // Load corner halo
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int halo_x = load_x + 1;
        int halo_y = load_y + 1;
        if (halo_x < width && halo_y < height) {
            shared_data[ty + 1][tx + 1] = input[halo_y * width + halo_x];
        } else {
            shared_data[ty + 1][tx + 1] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute Sobel if within valid region
    if (x < width && y < height) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators using shared memory
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                float pixel = shared_data[ty + ky][tx + kx];
                gx += c_sobel_x[ky * 3 + kx] * pixel;
                gy += c_sobel_y[ky * 3 + kx] * pixel;
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude));
    }
}

// Method 3: Sobel with gradient direction output
__global__ void sobel_edge_with_direction(const unsigned char* input, unsigned char* magnitude,
                                         unsigned char* direction, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int px = x + kx - 1;
                int py = y + ky - 1;
                int idx = py * width + px;
                
                float pixel = static_cast<float>(input[idx]);
                gx += c_sobel_x[ky * 3 + kx] * pixel;
                gy += c_sobel_y[ky * 3 + kx] * pixel;
            }
        }
        
        // Compute magnitude
        float mag = sqrtf(gx * gx + gy * gy);
        magnitude[y * width + x] = static_cast<unsigned char>(fminf(255.0f, mag));
        
        // Compute direction (quantized to 8 directions)
        float angle = atan2f(gy, gx);
        angle = angle * 180.0f / M_PI;
        if (angle < 0) angle += 180.0f;
        
        // Quantize to 8 directions (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5)
        int dir = static_cast<int>((angle + 11.25f) / 22.5f) % 8;
        direction[y * width + x] = static_cast<unsigned char>(dir * 32);
    } else if (x < width && y < height) {
        magnitude[y * width + x] = 0;
        if (direction) direction[y * width + x] = 0;
    }
}

// Method 4: Scharr operator (more accurate)
__global__ void scharr_edge_detection(const unsigned char* input, unsigned char* output,
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Scharr operators
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int px = x + kx - 1;
                int py = y + ky - 1;
                int idx = py * width + px;
                
                float pixel = static_cast<float>(input[idx]);
                gx += c_scharr_x[ky * 3 + kx] * pixel;
                gy += c_scharr_y[ky * 3 + kx] * pixel;
            }
        }
        
        // Compute gradient magnitude
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude / 16.0f));
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

// Method 5: Thresholded edge detection
__global__ void sobel_edge_threshold(const unsigned char* input, unsigned char* output,
                                    int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int px = x + kx - 1;
                int py = y + ky - 1;
                int idx = py * width + px;
                
                float pixel = static_cast<float>(input[idx]);
                gx += c_sobel_x[ky * 3 + kx] * pixel;
                gy += c_sobel_y[ky * 3 + kx] * pixel;
            }
        }
        
        // Compute gradient magnitude and threshold
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (magnitude > threshold) ? 255 : 0;
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

// Method 6: Non-maximum suppression for thin edges
__global__ void sobel_with_nms(const unsigned char* input, unsigned char* output,
                              int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int px = x + kx - 1;
                int py = y + ky - 1;
                int idx = py * width + px;
                
                float pixel = static_cast<float>(input[idx]);
                gx += c_sobel_x[ky * 3 + kx] * pixel;
                gy += c_sobel_y[ky * 3 + kx] * pixel;
            }
        }
        
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        if (magnitude > threshold) {
            // Determine gradient direction
            float angle = atan2f(gy, gx);
            angle = angle * 180.0f / M_PI;
            if (angle < 0) angle += 180.0f;
            
            // Non-maximum suppression
            float neighbor1 = 0.0f, neighbor2 = 0.0f;
            
            if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) {
                // Horizontal edge: check left and right
                neighbor1 = input[y * width + (x - 1)];
                neighbor2 = input[y * width + (x + 1)];
            } else if (angle >= 22.5f && angle < 67.5f) {
                // Diagonal edge: check diagonal neighbors
                neighbor1 = input[(y - 1) * width + (x + 1)];
                neighbor2 = input[(y + 1) * width + (x - 1)];
            } else if (angle >= 67.5f && angle < 112.5f) {
                // Vertical edge: check top and bottom
                neighbor1 = input[(y - 1) * width + x];
                neighbor2 = input[(y + 1) * width + x];
            } else if (angle >= 112.5f && angle < 157.5f) {
                // Diagonal edge: check other diagonal
                neighbor1 = input[(y - 1) * width + (x - 1)];
                neighbor2 = input[(y + 1) * width + (x + 1)];
            }
            
            // Suppress if not local maximum
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                output[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude));
            } else {
                output[y * width + x] = 0;
            }
        } else {
            output[y * width + x] = 0;
        }
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

// Host wrapper functions
extern "C" {

void launch_sobel_edge_naive(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_edge_naive<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_sobel_edge_shared(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, cudaStream_t stream = 0) {
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_edge_shared<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_sobel_edge_with_direction(const unsigned char* d_input, unsigned char* d_magnitude,
                                     unsigned char* d_direction, int width, int height,
                                     cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_edge_with_direction<<<grid_size, block_size, 0, stream>>>(d_input, d_magnitude, 
                                                                   d_direction, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_scharr_edge_detection(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    scharr_edge_detection<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_sobel_edge_threshold(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, float threshold, cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_edge_threshold<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                              width, height, threshold);
    CUDA_CHECK_KERNEL();
}

void launch_sobel_with_nms(const unsigned char* d_input, unsigned char* d_output,
                          int width, int height, float threshold, cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    sobel_with_nms<<<grid_size, block_size, 0, stream>>>(d_input, d_output, 
                                                        width, height, threshold);
    CUDA_CHECK_KERNEL();
}

// Utility function to choose best implementation
void launch_sobel_edge_auto(const unsigned char* d_input, unsigned char* d_output,
                           int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    
    if (total_pixels > 1920 * 1080) {
        // Large images: use shared memory version
        launch_sobel_edge_shared(d_input, d_output, width, height, stream);
    } else {
        // Smaller images: naive version is sufficient
        launch_sobel_edge_naive(d_input, d_output, width, height, stream);
    }
}

// Convenience function for complete edge detection pipeline
void launch_edge_detection_pipeline(const unsigned char* d_input, unsigned char* d_output,
                                   int width, int height, float threshold, bool use_nms,
                                   cudaStream_t stream = 0) {
    if (use_nms) {
        launch_sobel_with_nms(d_input, d_output, width, height, threshold, stream);
    } else {
        launch_sobel_edge_threshold(d_input, d_output, width, height, threshold, stream);
    }
}

} // extern "C"