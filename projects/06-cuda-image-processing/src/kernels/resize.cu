#include "cuda_utils.cuh"
#include <cuda_runtime.h>

// Texture reference for efficient sampling
texture<unsigned char, 2, cudaReadModeNormalizedFloat> tex_resize_input;

// Method 1: Nearest neighbor interpolation (naive)
__global__ void resize_nearest_neighbor(const unsigned char* input, unsigned char* output,
                                       int src_width, int src_height, 
                                       int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        // Calculate source coordinates
        float scale_x = static_cast<float>(src_width) / dst_width;
        float scale_y = static_cast<float>(src_height) / dst_height;
        
        int src_x = static_cast<int>(dst_x * scale_x);
        int src_y = static_cast<int>(dst_y * scale_y);
        
        // Clamp to source image bounds
        src_x = min(src_x, src_width - 1);
        src_y = min(src_y, src_height - 1);
        
        int src_idx = src_y * src_width + src_x;
        int dst_idx = dst_y * dst_width + dst_x;
        
        output[dst_idx] = input[src_idx];
    }
}

// Method 2: Bilinear interpolation
__global__ void resize_bilinear(const unsigned char* input, unsigned char* output,
                               int src_width, int src_height, 
                               int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        float scale_x = static_cast<float>(src_width - 1) / (dst_width - 1);
        float scale_y = static_cast<float>(src_height - 1) / (dst_height - 1);
        
        float src_x_f = dst_x * scale_x;
        float src_y_f = dst_y * scale_y;
        
        int src_x0 = static_cast<int>(floorf(src_x_f));
        int src_y0 = static_cast<int>(floorf(src_y_f));
        int src_x1 = min(src_x0 + 1, src_width - 1);
        int src_y1 = min(src_y0 + 1, src_height - 1);
        
        float wx = src_x_f - src_x0;
        float wy = src_y_f - src_y0;
        
        // Get four neighboring pixels
        unsigned char p00 = input[src_y0 * src_width + src_x0];
        unsigned char p01 = input[src_y0 * src_width + src_x1];
        unsigned char p10 = input[src_y1 * src_width + src_x0];
        unsigned char p11 = input[src_y1 * src_width + src_x1];
        
        // Bilinear interpolation
        float top = p00 * (1.0f - wx) + p01 * wx;
        float bottom = p10 * (1.0f - wx) + p11 * wx;
        float result = top * (1.0f - wy) + bottom * wy;
        
        int dst_idx = dst_y * dst_width + dst_x;
        output[dst_idx] = static_cast<unsigned char>(result);
    }
}

// Method 3: Texture memory with hardware interpolation
__global__ void resize_texture(unsigned char* output, int dst_width, int dst_height,
                              float scale_x, float scale_y) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        // Calculate normalized texture coordinates
        float tex_x = (dst_x + 0.5f) / scale_x;
        float tex_y = (dst_y + 0.5f) / scale_y;
        
        // Hardware-accelerated bilinear interpolation
        float texel = tex2D(tex_resize_input, tex_x, tex_y);
        
        int dst_idx = dst_y * dst_width + dst_x;
        output[dst_idx] = static_cast<unsigned char>(texel * 255.0f);
    }
}

// Method 4: RGB image resizing with bilinear interpolation
__global__ void resize_rgb_bilinear(const unsigned char* input, unsigned char* output,
                                   int src_width, int src_height,
                                   int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        float scale_x = static_cast<float>(src_width - 1) / (dst_width - 1);
        float scale_y = static_cast<float>(src_height - 1) / (dst_height - 1);
        
        float src_x_f = dst_x * scale_x;
        float src_y_f = dst_y * scale_y;
        
        int src_x0 = static_cast<int>(floorf(src_x_f));
        int src_y0 = static_cast<int>(floorf(src_y_f));
        int src_x1 = min(src_x0 + 1, src_width - 1);
        int src_y1 = min(src_y0 + 1, src_height - 1);
        
        float wx = src_x_f - src_x0;
        float wy = src_y_f - src_y0;
        
        // Process all three channels (RGB)
        for (int c = 0; c < 3; c++) {
            // Get four neighboring pixels for this channel
            unsigned char p00 = input[(src_y0 * src_width + src_x0) * 3 + c];
            unsigned char p01 = input[(src_y0 * src_width + src_x1) * 3 + c];
            unsigned char p10 = input[(src_y1 * src_width + src_x0) * 3 + c];
            unsigned char p11 = input[(src_y1 * src_width + src_x1) * 3 + c];
            
            // Bilinear interpolation for this channel
            float top = p00 * (1.0f - wx) + p01 * wx;
            float bottom = p10 * (1.0f - wx) + p11 * wx;
            float result = top * (1.0f - wy) + bottom * wy;
            
            int dst_idx = (dst_y * dst_width + dst_x) * 3 + c;
            output[dst_idx] = static_cast<unsigned char>(result);
        }
    }
}

// Method 5: Bicubic interpolation (higher quality)
__device__ float cubic_interpolate(float p0, float p1, float p2, float p3, float t) {
    float a0 = p3 - p2 - p0 + p1;
    float a1 = p0 - p1 - a0;
    float a2 = p2 - p0;
    float a3 = p1;
    
    return a0 * t * t * t + a1 * t * t + a2 * t + a3;
}

__global__ void resize_bicubic(const unsigned char* input, unsigned char* output,
                              int src_width, int src_height,
                              int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        float scale_x = static_cast<float>(src_width) / dst_width;
        float scale_y = static_cast<float>(src_height) / dst_height;
        
        float src_x_f = dst_x * scale_x;
        float src_y_f = dst_y * scale_y;
        
        int src_x = static_cast<int>(floorf(src_x_f));
        int src_y = static_cast<int>(floorf(src_y_f));
        
        float dx = src_x_f - src_x;
        float dy = src_y_f - src_y;
        
        // Sample 4x4 neighborhood for bicubic interpolation
        float temp[4];
        for (int j = 0; j < 4; j++) {
            int y_coord = max(0, min(src_height - 1, src_y - 1 + j));
            float p[4];
            for (int i = 0; i < 4; i++) {
                int x_coord = max(0, min(src_width - 1, src_x - 1 + i));
                p[i] = static_cast<float>(input[y_coord * src_width + x_coord]);
            }
            temp[j] = cubic_interpolate(p[0], p[1], p[2], p[3], dx);
        }
        
        float result = cubic_interpolate(temp[0], temp[1], temp[2], temp[3], dy);
        
        int dst_idx = dst_y * dst_width + dst_x;
        output[dst_idx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, result)));
    }
}

// Method 6: Lanczos resampling (high quality downscaling)
__device__ float lanczos_kernel(float x, int a = 3) {
    if (x == 0.0f) return 1.0f;
    if (x >= a || x <= -a) return 0.0f;
    
    float pi_x = M_PI * x;
    return (a * sinf(pi_x) * sinf(pi_x / a)) / (pi_x * pi_x);
}

__global__ void resize_lanczos(const unsigned char* input, unsigned char* output,
                              int src_width, int src_height,
                              int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x < dst_width && dst_y < dst_height) {
        float scale_x = static_cast<float>(src_width) / dst_width;
        float scale_y = static_cast<float>(src_height) / dst_height;
        
        float src_x_f = dst_x * scale_x;
        float src_y_f = dst_y * scale_y;
        
        int radius = 3;
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int j = -radius; j <= radius; j++) {
            for (int i = -radius; i <= radius; i++) {
                int src_x = static_cast<int>(floorf(src_x_f)) + i;
                int src_y = static_cast<int>(floorf(src_y_f)) + j;
                
                if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height) {
                    float dx = src_x_f - src_x;
                    float dy = src_y_f - src_y;
                    
                    float weight = lanczos_kernel(dx) * lanczos_kernel(dy);
                    sum += weight * input[src_y * src_width + src_x];
                    weight_sum += weight;
                }
            }
        }
        
        float result = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
        int dst_idx = dst_y * dst_width + dst_x;
        output[dst_idx] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, result)));
    }
}

// Host wrapper functions
extern "C" {

void launch_resize_nearest_neighbor(const unsigned char* d_input, unsigned char* d_output,
                                   int src_width, int src_height,
                                   int dst_width, int dst_height,
                                   cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_nearest_neighbor<<<grid_size, block_size, 0, stream>>>(d_input, d_output,
                                                                 src_width, src_height,
                                                                 dst_width, dst_height);
    CUDA_CHECK_KERNEL();
}

void launch_resize_bilinear(const unsigned char* d_input, unsigned char* d_output,
                           int src_width, int src_height,
                           int dst_width, int dst_height,
                           cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_bilinear<<<grid_size, block_size, 0, stream>>>(d_input, d_output,
                                                         src_width, src_height,
                                                         dst_width, dst_height);
    CUDA_CHECK_KERNEL();
}

void launch_resize_texture(const unsigned char* d_input, unsigned char* d_output,
                          int src_width, int src_height,
                          int dst_width, int dst_height,
                          cudaStream_t stream = 0) {
    // Bind texture
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaBindTexture2D(nullptr, tex_resize_input, d_input, 
                                 channel_desc, src_width, src_height, src_width));
    
    // Set texture parameters
    tex_resize_input.addressMode[0] = cudaAddressModeClamp;
    tex_resize_input.addressMode[1] = cudaAddressModeClamp;
    tex_resize_input.filterMode = cudaFilterModeLinear;
    tex_resize_input.normalized = false;
    
    float scale_x = static_cast<float>(dst_width) / src_width;
    float scale_y = static_cast<float>(dst_height) / src_height;
    
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_texture<<<grid_size, block_size, 0, stream>>>(d_output, dst_width, dst_height,
                                                        scale_x, scale_y);
    CUDA_CHECK_KERNEL();
    
    // Unbind texture
    CUDA_CHECK(cudaUnbindTexture(tex_resize_input));
}

void launch_resize_rgb_bilinear(const unsigned char* d_input, unsigned char* d_output,
                               int src_width, int src_height,
                               int dst_width, int dst_height,
                               cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_rgb_bilinear<<<grid_size, block_size, 0, stream>>>(d_input, d_output,
                                                             src_width, src_height,
                                                             dst_width, dst_height);
    CUDA_CHECK_KERNEL();
}

void launch_resize_bicubic(const unsigned char* d_input, unsigned char* d_output,
                          int src_width, int src_height,
                          int dst_width, int dst_height,
                          cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_bicubic<<<grid_size, block_size, 0, stream>>>(d_input, d_output,
                                                        src_width, src_height,
                                                        dst_width, dst_height);
    CUDA_CHECK_KERNEL();
}

void launch_resize_lanczos(const unsigned char* d_input, unsigned char* d_output,
                          int src_width, int src_height,
                          int dst_width, int dst_height,
                          cudaStream_t stream = 0) {
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                   (dst_height + block_size.y - 1) / block_size.y);
    
    resize_lanczos<<<grid_size, block_size, 0, stream>>>(d_input, d_output,
                                                        src_width, src_height,
                                                        dst_width, dst_height);
    CUDA_CHECK_KERNEL();
}

// Utility function to choose best implementation based on scale factor
void launch_resize_auto(const unsigned char* d_input, unsigned char* d_output,
                       int src_width, int src_height,
                       int dst_width, int dst_height,
                       cudaStream_t stream = 0) {
    float scale_x = static_cast<float>(dst_width) / src_width;
    float scale_y = static_cast<float>(dst_height) / src_height;
    float scale_factor = (scale_x + scale_y) / 2.0f;
    
    if (scale_factor > 2.0f) {
        // Upscaling: use bicubic for better quality
        launch_resize_bicubic(d_input, d_output, src_width, src_height,
                            dst_width, dst_height, stream);
    } else if (scale_factor < 0.5f) {
        // Significant downscaling: use Lanczos for anti-aliasing
        launch_resize_lanczos(d_input, d_output, src_width, src_height,
                            dst_width, dst_height, stream);
    } else {
        // Moderate scaling: use bilinear (good balance of quality/speed)
        launch_resize_bilinear(d_input, d_output, src_width, src_height,
                             dst_width, dst_height, stream);
    }
}

// Convenience function for common scale factors
void launch_resize_by_factor(const unsigned char* d_input, unsigned char* d_output,
                            int width, int height, float scale_factor,
                            cudaStream_t stream = 0) {
    int dst_width = static_cast<int>(width * scale_factor);
    int dst_height = static_cast<int>(height * scale_factor);
    
    launch_resize_auto(d_input, d_output, width, height, dst_width, dst_height, stream);
}

} // extern "C"