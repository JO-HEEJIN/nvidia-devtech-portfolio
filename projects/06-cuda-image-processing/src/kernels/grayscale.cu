#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// Grayscale conversion coefficients (ITU-R BT.709 standard)
#define R_COEFF 0.299f
#define G_COEFF 0.587f
#define B_COEFF 0.114f

// Simple grayscale conversion: one thread per pixel
__global__ void grayscale_simple(const unsigned char* input, unsigned char* output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;
        
        // Calculate luminance using ITU-R BT.709 coefficients
        float gray = R_COEFF * input[rgb_idx] +     // R
                     G_COEFF * input[rgb_idx + 1] + // G
                     B_COEFF * input[rgb_idx + 2];  // B
        
        output[idx] = static_cast<unsigned char>(gray);
    }
}

// Optimized version using vectorized loads (uchar4)
__global__ void grayscale_vectorized(const unsigned char* input, unsigned char* output,
                                    int width, int height) {
    int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;
        
        // Check bounds for vectorized access
        if (x + 3 < width) {
            // Load 4 pixels worth of RGB data (12 bytes)
            uchar4 rgb1 = *reinterpret_cast<const uchar4*>(&input[rgb_idx]);
            uchar4 rgb2 = *reinterpret_cast<const uchar4*>(&input[rgb_idx + 4]);
            uchar4 rgb3 = *reinterpret_cast<const uchar4*>(&input[rgb_idx + 8]);
            
            // Extract RGB values for 4 pixels
            unsigned char r1 = rgb1.x, g1 = rgb1.y, b1 = rgb1.z;
            unsigned char r2 = rgb1.w, g2 = rgb2.x, b2 = rgb2.y;
            unsigned char r3 = rgb2.z, g3 = rgb2.w, b3 = rgb3.x;
            unsigned char r4 = rgb3.y, g4 = rgb3.z, b4 = rgb3.w;
            
            // Calculate grayscale for 4 pixels
            unsigned char gray1 = static_cast<unsigned char>(R_COEFF * r1 + G_COEFF * g1 + B_COEFF * b1);
            unsigned char gray2 = static_cast<unsigned char>(R_COEFF * r2 + G_COEFF * g2 + B_COEFF * b2);
            unsigned char gray3 = static_cast<unsigned char>(R_COEFF * r3 + G_COEFF * g3 + B_COEFF * b3);
            unsigned char gray4 = static_cast<unsigned char>(R_COEFF * r4 + G_COEFF * g4 + B_COEFF * b4);
            
            // Store 4 pixels as a single uchar4
            uchar4 result = make_uchar4(gray1, gray2, gray3, gray4);
            *reinterpret_cast<uchar4*>(&output[idx]) = result;
        } else {
            // Handle remaining pixels individually
            for (int i = 0; i < 4 && x + i < width; i++) {
                int pixel_idx = idx + i;
                int pixel_rgb_idx = pixel_idx * 3;
                
                float gray = R_COEFF * input[pixel_rgb_idx] +
                           G_COEFF * input[pixel_rgb_idx + 1] +
                           B_COEFF * input[pixel_rgb_idx + 2];
                
                output[pixel_idx] = static_cast<unsigned char>(gray);
            }
        }
    }
}

// Coalesced memory access version
__global__ void grayscale_coalesced(const unsigned char* input, unsigned char* output,
                                   int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        int rgb_idx = idx * 3;
        
        // Coalesced read of RGB values
        unsigned char r = input[rgb_idx];
        unsigned char g = input[rgb_idx + 1];
        unsigned char b = input[rgb_idx + 2];
        
        // Calculate grayscale value
        float gray = R_COEFF * r + G_COEFF * g + B_COEFF * b;
        
        // Coalesced write
        output[idx] = static_cast<unsigned char>(gray);
    }
}

// Host wrapper functions
extern "C" {

void launch_grayscale_simple(const unsigned char* d_input, unsigned char* d_output,
                            int width, int height, cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    grayscale_simple<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_grayscale_vectorized(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, cudaStream_t stream = 0) {
    // Process 4 pixels per thread in x direction
    dim3 block_size(16, 16);
    dim3 grid_size((width / 4 + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    
    grayscale_vectorized<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_grayscale_coalesced(const unsigned char* d_input, unsigned char* d_output,
                               int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    int block_size = 256;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    grayscale_coalesced<<<grid_size, block_size, 0, stream>>>(d_input, d_output, width, height);
    CUDA_CHECK_KERNEL();
}

// Utility function to choose best implementation based on image size
void launch_grayscale_auto(const unsigned char* d_input, unsigned char* d_output,
                          int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    
    // Choose implementation based on image size and characteristics
    if (total_pixels > 1920 * 1080) {
        // Large images: use coalesced version for best memory throughput
        launch_grayscale_coalesced(d_input, d_output, width, height, stream);
    } else if (width % 4 == 0 && total_pixels > 720 * 480) {
        // Medium images with width divisible by 4: use vectorized
        launch_grayscale_vectorized(d_input, d_output, width, height, stream);
    } else {
        // Small images or odd widths: use simple version
        launch_grayscale_simple(d_input, d_output, width, height, stream);
    }
}

} // extern "C"