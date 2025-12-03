#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <math.h>

#define MAX_KERNEL_RADIUS 10
#define TILE_SIZE 16
#define SHARED_SIZE (TILE_SIZE + 2 * MAX_KERNEL_RADIUS)

// Constant memory for Gaussian kernel weights
__constant__ float c_gaussian_kernel[2 * MAX_KERNEL_RADIUS + 1][2 * MAX_KERNEL_RADIUS + 1];
__constant__ float c_gaussian_kernel_1d[2 * MAX_KERNEL_RADIUS + 1];

// Generate Gaussian kernel weights on host
void generate_gaussian_kernel(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    int size = 2 * radius + 1;
    
    // Generate 2D Gaussian kernel
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel[(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

void generate_gaussian_kernel_1d(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    int size = 2 * radius + 1;
    
    // Generate 1D Gaussian kernel
    for (int x = -radius; x <= radius; x++) {
        float value = expf(-(x * x) / (2.0f * sigma * sigma));
        kernel[x + radius] = value;
        sum += value;
    }
    
    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

// Method 1: Naive global memory implementation
__global__ void gaussian_blur_naive(const unsigned char* input, unsigned char* output,
                                   int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int px = x + kx;
                int py = y + ky;
                
                // Clamp to image boundaries
                px = max(0, min(width - 1, px));
                py = max(0, min(height - 1, py));
                
                float weight = c_gaussian_kernel[ky + radius][kx + radius];
                sum += weight * input[py * width + px];
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// Method 2: Shared memory with halo regions
__global__ void gaussian_blur_shared(const unsigned char* input, unsigned char* output,
                                    int width, int height, int radius) {
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Load main tile data
    int load_x = x - radius;
    int load_y = y - radius;
    
    // Load data with halo region
    for (int dy = 0; dy < SHARED_SIZE; dy += blockDim.y) {
        for (int dx = 0; dx < SHARED_SIZE; dx += blockDim.x) {
            int shared_x = tx + dx;
            int shared_y = ty + dy;
            
            if (shared_x < SHARED_SIZE && shared_y < SHARED_SIZE) {
                int global_x = load_x + shared_x;
                int global_y = load_y + shared_y;
                
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
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int shared_x = tx + radius + kx;
                int shared_y = ty + radius + ky;
                
                float weight = c_gaussian_kernel[ky + radius][kx + radius];
                sum += weight * shared_data[shared_y][shared_x];
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// Method 3: Separable filter (horizontal pass)
__global__ void gaussian_blur_horizontal(const unsigned char* input, unsigned char* output,
                                        int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int kx = -radius; kx <= radius; kx++) {
            int px = x + kx;
            px = max(0, min(width - 1, px));
            
            float weight = c_gaussian_kernel_1d[kx + radius];
            sum += weight * input[y * width + px];
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// Method 3: Separable filter (vertical pass)
__global__ void gaussian_blur_vertical(const unsigned char* input, unsigned char* output,
                                      int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            int py = y + ky;
            py = max(0, min(height - 1, py));
            
            float weight = c_gaussian_kernel_1d[ky + radius];
            sum += weight * input[py * width + x];
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

// Method 4: Texture memory implementation
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_input;

__global__ void gaussian_blur_texture(unsigned char* output, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                float px = x + kx + 0.5f;
                float py = y + ky + 0.5f;
                
                float weight = c_gaussian_kernel[ky + radius][kx + radius];
                float texel = tex2D(tex_input, px, py);
                sum += weight * texel;
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum * 255.0f);
    }
}

// Host wrapper functions
extern "C" {

void setup_gaussian_kernel(int radius, float sigma) {
    if (radius > MAX_KERNEL_RADIUS) {
        radius = MAX_KERNEL_RADIUS;
    }
    
    int size = 2 * radius + 1;
    float* h_kernel_2d = new float[size * size];
    float* h_kernel_1d = new float[size];
    
    generate_gaussian_kernel(h_kernel_2d, radius, sigma);
    generate_gaussian_kernel_1d(h_kernel_1d, radius, sigma);
    
    // Copy to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian_kernel, h_kernel_2d, 
                                  size * size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian_kernel_1d, h_kernel_1d, 
                                  size * sizeof(float)));
    
    delete[] h_kernel_2d;
    delete[] h_kernel_1d;
}

void launch_gaussian_blur_naive(const unsigned char* d_input, unsigned char* d_output,
                               int width, int height, int radius, float sigma,
                               cudaStream_t stream = 0) {
    setup_gaussian_kernel(radius, sigma);
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    gaussian_blur_naive<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height, radius);
    CUDA_CHECK_KERNEL();
}

void launch_gaussian_blur_shared(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, int radius, float sigma,
                                cudaStream_t stream = 0) {
    setup_gaussian_kernel(radius, sigma);
    
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    gaussian_blur_shared<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height, radius);
    CUDA_CHECK_KERNEL();
}

void launch_gaussian_blur_separable(const unsigned char* d_input, unsigned char* d_output,
                                   int width, int height, int radius, float sigma,
                                   unsigned char* d_temp = nullptr, cudaStream_t stream = 0) {
    setup_gaussian_kernel(radius, sigma);
    
    // Allocate temporary buffer if not provided
    unsigned char* temp_buffer = d_temp;
    bool allocated_temp = false;
    if (!temp_buffer) {
        CUDA_CHECK(cudaMalloc(&temp_buffer, width * height * sizeof(unsigned char)));
        allocated_temp = true;
    }
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    // Horizontal pass
    gaussian_blur_horizontal<<<grid_size, block_size, 0, stream>>>(d_input, temp_buffer, 
                                                                   width, height, radius);
    CUDA_CHECK_KERNEL();
    
    // Vertical pass
    gaussian_blur_vertical<<<grid_size, block_size, 0, stream>>>(temp_buffer, d_output, 
                                                                 width, height, radius);
    CUDA_CHECK_KERNEL();
    
    if (allocated_temp) {
        CUDA_CHECK(cudaFree(temp_buffer));
    }
}

void launch_gaussian_blur_texture(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, int radius, float sigma,
                                 cudaStream_t stream = 0) {
    setup_gaussian_kernel(radius, sigma);
    
    // Bind texture
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaBindTexture2D(nullptr, tex_input, d_input, channel_desc, width, height, width));
    
    // Set texture parameters
    tex_input.addressMode[0] = cudaAddressModeClamp;
    tex_input.addressMode[1] = cudaAddressModeClamp;
    tex_input.filterMode = cudaFilterModeLinear;
    tex_input.normalized = false;
    
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    gaussian_blur_texture<<<grid_size, block_size, 0, stream>>>(d_output, width, height, radius);
    CUDA_CHECK_KERNEL();
    
    // Unbind texture
    CUDA_CHECK(cudaUnbindTexture(tex_input));
}

// Utility function to choose best implementation
void launch_gaussian_blur_auto(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, int radius, float sigma,
                              cudaStream_t stream = 0) {
    if (radius > MAX_KERNEL_RADIUS) {
        radius = MAX_KERNEL_RADIUS;
    }
    
    int total_pixels = width * height;
    
    if (radius >= 5 && total_pixels > 1920 * 1080) {
        // Large kernel and image: use separable filter
        launch_gaussian_blur_separable(d_input, d_output, width, height, radius, sigma, nullptr, stream);
    } else if (total_pixels > 720 * 480) {
        // Medium image: use shared memory
        launch_gaussian_blur_shared(d_input, d_output, width, height, radius, sigma, stream);
    } else {
        // Small image: naive implementation may be sufficient
        launch_gaussian_blur_naive(d_input, d_output, width, height, radius, sigma, stream);
    }
}

} // extern "C"