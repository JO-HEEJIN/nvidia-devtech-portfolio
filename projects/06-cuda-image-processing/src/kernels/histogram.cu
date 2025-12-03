#include "cuda_utils.cuh"
#include <cuda_runtime.h>

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256

// Method 1: Naive atomic operations (global memory)
__global__ void histogram_atomic_global(const unsigned char* input, unsigned int* histogram,
                                       int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx < total_pixels) {
        unsigned char pixel_value = input[idx];
        atomicAdd(&histogram[pixel_value], 1);
    }
}

// Method 2: Shared memory privatization to reduce contention
__global__ void histogram_shared_privatization(const unsigned char* input, unsigned int* histogram,
                                              int width, int height) {
    __shared__ unsigned int shared_hist[HISTOGRAM_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    // Initialize shared histogram
    if (tid < HISTOGRAM_SIZE) {
        shared_hist[tid] = 0;
    }
    __syncthreads();
    
    // Process multiple pixels per thread to increase work per thread
    int elements_per_thread = (total_pixels + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int start_idx = idx * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, total_pixels);
    
    // Accumulate histogram in shared memory
    for (int i = start_idx; i < end_idx; i++) {
        if (i < total_pixels) {
            unsigned char pixel_value = input[i];
            atomicAdd(&shared_hist[pixel_value], 1);
        }
    }
    __syncthreads();
    
    // Merge shared histogram to global histogram
    if (tid < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[tid], shared_hist[tid]);
    }
}

// Method 3: Reduction-based approach (more complex but can be faster for large images)
__global__ void histogram_reduction_kernel(const unsigned char* input, unsigned int* partial_histograms,
                                          int width, int height, int hist_offset) {
    __shared__ unsigned int shared_hist[HISTOGRAM_SIZE];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    int total_pixels = width * height;
    
    // Initialize shared histogram
    if (tid < HISTOGRAM_SIZE) {
        shared_hist[tid] = 0;
    }
    __syncthreads();
    
    // Process pixels
    if (idx < total_pixels) {
        unsigned char pixel_value = input[idx];
        atomicAdd(&shared_hist[pixel_value], 1);
    }
    __syncthreads();
    
    // Store partial histogram for this block
    if (tid < HISTOGRAM_SIZE) {
        partial_histograms[bid * HISTOGRAM_SIZE + tid] = shared_hist[tid];
    }
}

__global__ void histogram_reduction_merge(unsigned int* partial_histograms, unsigned int* final_histogram,
                                         int num_blocks) {
    int tid = threadIdx.x;
    
    if (tid < HISTOGRAM_SIZE) {
        unsigned int sum = 0;
        for (int i = 0; i < num_blocks; i++) {
            sum += partial_histograms[i * HISTOGRAM_SIZE + tid];
        }
        final_histogram[tid] = sum;
    }
}

// Method 4: Optimized version with warp-level primitives (for modern GPUs)
__global__ void histogram_warp_optimized(const unsigned char* input, unsigned int* histogram,
                                        int width, int height) {
    __shared__ unsigned int shared_hist[HISTOGRAM_SIZE];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    // Initialize shared histogram (only first warp)
    if (warp_id == 0 && lane_id < HISTOGRAM_SIZE) {
        shared_hist[lane_id] = 0;
    }
    __syncthreads();
    
    // Process pixels with stride
    for (int i = idx; i < total_pixels; i += gridDim.x * blockDim.x) {
        unsigned char pixel_value = input[i];
        atomicAdd(&shared_hist[pixel_value], 1);
    }
    __syncthreads();
    
    // Merge to global histogram
    if (warp_id == 0 && lane_id < HISTOGRAM_SIZE) {
        atomicAdd(&histogram[lane_id], shared_hist[lane_id]);
    }
}

// Host wrapper functions
extern "C" {

void launch_histogram_atomic_global(const unsigned char* d_input, unsigned int* d_histogram,
                                   int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    int block_size = BLOCK_SIZE;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    // Clear histogram
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int), stream));
    
    histogram_atomic_global<<<grid_size, block_size, 0, stream>>>(d_input, d_histogram, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_histogram_shared_privatization(const unsigned char* d_input, unsigned int* d_histogram,
                                          int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    int block_size = BLOCK_SIZE;
    int grid_size = min((total_pixels + block_size - 1) / block_size, 65535);
    
    // Clear histogram
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int), stream));
    
    histogram_shared_privatization<<<grid_size, block_size, 0, stream>>>(d_input, d_histogram, width, height);
    CUDA_CHECK_KERNEL();
}

void launch_histogram_reduction(const unsigned char* d_input, unsigned int* d_histogram,
                               int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    int block_size = BLOCK_SIZE;
    int grid_size = (total_pixels + block_size - 1) / block_size;
    
    // Allocate temporary storage for partial histograms
    unsigned int* d_partial_histograms;
    CUDA_CHECK(cudaMalloc(&d_partial_histograms, grid_size * HISTOGRAM_SIZE * sizeof(unsigned int)));
    
    // Clear final histogram
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int), stream));
    
    // Phase 1: Compute partial histograms
    histogram_reduction_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_partial_histograms, 
                                                                     width, height, 0);
    CUDA_CHECK_KERNEL();
    
    // Phase 2: Merge partial histograms
    int merge_block_size = min(HISTOGRAM_SIZE, 256);
    histogram_reduction_merge<<<1, merge_block_size, 0, stream>>>(d_partial_histograms, d_histogram, grid_size);
    CUDA_CHECK_KERNEL();
    
    CUDA_CHECK(cudaFree(d_partial_histograms));
}

void launch_histogram_warp_optimized(const unsigned char* d_input, unsigned int* d_histogram,
                                    int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    int block_size = BLOCK_SIZE;
    int grid_size = min((total_pixels + block_size - 1) / block_size, 512); // Limit grid size for efficiency
    
    // Clear histogram
    CUDA_CHECK(cudaMemsetAsync(d_histogram, 0, HISTOGRAM_SIZE * sizeof(unsigned int), stream));
    
    histogram_warp_optimized<<<grid_size, block_size, 0, stream>>>(d_input, d_histogram, width, height);
    CUDA_CHECK_KERNEL();
}

// Utility function to choose best implementation based on image size and GPU architecture
void launch_histogram_auto(const unsigned char* d_input, unsigned int* d_histogram,
                          int width, int height, cudaStream_t stream = 0) {
    int total_pixels = width * height;
    
    // Get device properties to make informed decision
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    if (prop.major >= 7 && total_pixels > 1920 * 1080) {
        // Modern GPU with large image: use warp-optimized version
        launch_histogram_warp_optimized(d_input, d_histogram, width, height, stream);
    } else if (total_pixels > 720 * 480) {
        // Medium to large image: use shared memory privatization
        launch_histogram_shared_privatization(d_input, d_histogram, width, height, stream);
    } else {
        // Small image: simple atomic operations may be sufficient
        launch_histogram_atomic_global(d_input, d_histogram, width, height, stream);
    }
}

// Utility function to compute histogram statistics
void compute_histogram_stats(const unsigned int* h_histogram, int total_pixels,
                            float* mean, float* std_dev, int* min_val, int* max_val) {
    *mean = 0.0f;
    *std_dev = 0.0f;
    *min_val = -1;
    *max_val = -1;
    
    // Find min and max non-zero values
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_histogram[i] > 0) {
            if (*min_val == -1) *min_val = i;
            *max_val = i;
        }
    }
    
    // Compute mean
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        *mean += i * h_histogram[i];
    }
    *mean /= total_pixels;
    
    // Compute standard deviation
    float variance = 0.0f;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        float diff = i - *mean;
        variance += diff * diff * h_histogram[i];
    }
    variance /= total_pixels;
    *std_dev = sqrtf(variance);
}

} // extern "C"