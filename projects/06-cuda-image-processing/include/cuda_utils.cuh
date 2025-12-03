#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA operation failed"); \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA kernel launch failed"); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// Simple timer class using CUDA events
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        CUDA_CHECK(cudaEventRecord(start));
    }
    
    float stopTimer() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};

// Memory allocation helpers
template<typename T>
T* cuda_malloc(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
void cuda_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Memory copy helpers
template<typename T>
void cuda_memcpy_h2d(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cuda_memcpy_d2h(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void cuda_memcpy_d2d(T* dst, const T* src, size_t count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// Thread block dimension helpers
inline dim3 compute_grid_size(int width, int height, int block_size = 16) {
    dim3 grid;
    grid.x = (width + block_size - 1) / block_size;
    grid.y = (height + block_size - 1) / block_size;
    grid.z = 1;
    return grid;
}

inline dim3 compute_block_size(int block_size = 16) {
    return dim3(block_size, block_size, 1);
}

// Device information helper
inline void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
}

// Texture memory helpers
template<typename T>
cudaTextureObject_t create_texture_object(T* d_data, int width, int height) {
    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(T);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    
    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    
    // Create texture object
    cudaTextureObject_t texObj;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    
    return texObj;
}

inline void destroy_texture_object(cudaTextureObject_t texObj) {
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
}

#endif // CUDA_UTILS_CUH