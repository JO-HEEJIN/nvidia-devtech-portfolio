#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "matrix.h"

// CUDA timer class using events
class CudaTimer {
private:
    cudaEvent_t start, stop;
    float elapsed_time;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        elapsed_time = 0.0f;
    }
    
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    void startTimer() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }
    
    void stopTimer() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    }
    
    float getElapsedTime() {
        return elapsed_time;
    }
    
    void reset() {
        elapsed_time = 0.0f;
    }
};

// Allocate and initialize matrices on host
void allocateHostMatrices(Matrix& A, Matrix& B, Matrix& C, int N) {
    A = allocateMatrix(N, N);
    B = allocateMatrix(N, N);
    C = allocateMatrix(N, N);
    
    // Initialize with random values
    srand(42);  // Fixed seed for reproducibility
    randomizeMatrix(A, 0.0f, 1.0f);
    randomizeMatrix(B, 0.0f, 1.0f);
    initializeMatrix(C, 0.0f);
}

// Allocate matrices on device
void allocateDeviceMatrices(Matrix& d_A, Matrix& d_B, Matrix& d_C, int N) {
    d_A = allocateDeviceMatrix(N, N);
    d_B = allocateDeviceMatrix(N, N);
    d_C = allocateDeviceMatrix(N, N);
    
    // Initialize C to zero
    size_t size = N * N * sizeof(float);
    CUDA_CHECK(cudaMemset(d_C.elements, 0, size));
}

// Transfer matrices from host to device
void transferToDevice(const Matrix h_A, const Matrix h_B, 
                     Matrix d_A, Matrix d_B) {
    copyMatrixHostToDevice(h_A, d_A);
    copyMatrixHostToDevice(h_B, d_B);
}

// Transfer result from device to host
void transferFromDevice(Matrix d_C, Matrix h_C) {
    copyMatrixDeviceToHost(d_C, h_C);
}

// Free all device matrices
void freeDeviceMatrices(Matrix d_A, Matrix d_B, Matrix d_C) {
    freeDeviceMatrix(d_A);
    freeDeviceMatrix(d_B);
    freeDeviceMatrix(d_C);
}

// Free all host matrices
void freeHostMatrices(Matrix A, Matrix B, Matrix C) {
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);
}

// Load CPU result for verification
bool loadCPUResult(Matrix& cpu_result, int N) {
    FILE* fp = fopen("cpu_result.bin", "rb");
    if (!fp) {
        printf("Warning: CPU result file not found. Skipping verification.\n");
        return false;
    }
    
    cpu_result = allocateMatrix(N, N);
    size_t read = fread(cpu_result.elements, sizeof(float), N * N, fp);
    fclose(fp);
    
    if (read != (size_t)(N * N)) {
        printf("Warning: CPU result file size mismatch. Skipping verification.\n");
        freeMatrix(cpu_result);
        return false;
    }
    
    return true;
}

// Compare GPU result with CPU result
bool verifyWithCPU(Matrix gpu_result, int N) {
    Matrix cpu_result;
    if (!loadCPUResult(cpu_result, N)) {
        return false;
    }
    
    bool passed = verifyResult(cpu_result, gpu_result);
    freeMatrix(cpu_result);
    return passed;
}

// Benchmark helper function
float benchmarkKernel(void (*kernel)(Matrix, Matrix, Matrix, dim3, dim3),
                      Matrix d_A, Matrix d_B, Matrix d_C,
                      dim3 gridDim, dim3 blockDim,
                      int iterations = 10, int warmup = 5) {
    
    CudaTimer timer;
    
    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        kernel(d_A, d_B, d_C, gridDim, blockDim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    timer.startTimer();
    for (int i = 0; i < iterations; i++) {
        kernel(d_A, d_B, d_C, gridDim, blockDim);
    }
    timer.stopTimer();
    
    return timer.getElapsedTime() / iterations;
}

// Print performance summary
void printPerformanceSummary(const char* kernelName, int N, float time_ms) {
    float gflops = calculateGFLOPS(N, time_ms);
    
    // Memory operations: read each element of A once, each element of B N times,
    // write each element of C once
    size_t memoryOps = N * N * sizeof(float) +  // Read A
                       N * N * N * sizeof(float) +  // Read B (N times)
                       N * N * sizeof(float);  // Write C
    float bandwidth = calculateBandwidth(memoryOps, time_ms);
    
    printf("\n%s Performance:\n", kernelName);
    printf("  Time: %.3f ms\n", time_ms);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    printf("  Effective Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Calculate arithmetic intensity
    float ops = 2.0f * N * N * N;
    float bytes = memoryOps;
    float arithmetic_intensity = ops / bytes;
    printf("  Arithmetic Intensity: %.2f FLOP/byte\n", arithmetic_intensity);
}

// Get optimal block size for the current device
dim3 getOptimalBlockSize(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Default to 16x16 for most GPUs
    int blockSize = 16;
    
    // Adjust based on compute capability
    if (prop.major >= 7) {  // Volta and newer
        blockSize = 32;  // Can use larger tiles
    }
    
    // Ensure we don't exceed limits
    if (blockSize * blockSize > prop.maxThreadsPerBlock) {
        blockSize = 16;
    }
    
    return dim3(blockSize, blockSize);
}

// Memory pool for optimized allocations (CUDA 11.2+)
class CudaMemoryPool {
private:
    static constexpr size_t POOL_SIZE = 1024 * 1024 * 1024;  // 1GB pool
    void* pool_ptr;
    size_t current_offset;
    bool initialized;
    
public:
    CudaMemoryPool() : pool_ptr(nullptr), current_offset(0), initialized(false) {
        #if CUDART_VERSION >= 11020
        // Use CUDA memory pool if available
        cudaMemPool_t mempool;
        cudaDeviceGetDefaultMemPool(&mempool, 0);
        
        size_t threshold = POOL_SIZE;
        cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
        initialized = true;
        #endif
    }
    
    void* allocate(size_t size) {
        if (!initialized) {
            void* ptr;
            CUDA_CHECK(cudaMalloc(&ptr, size));
            return ptr;
        }
        
        // Simple linear allocation
        if (current_offset + size > POOL_SIZE) {
            printf("Memory pool exhausted!\n");
            return nullptr;
        }
        
        void* ptr = (char*)pool_ptr + current_offset;
        current_offset += size;
        return ptr;
    }
    
    void reset() {
        current_offset = 0;
    }
    
    ~CudaMemoryPool() {
        if (pool_ptr) {
            cudaFree(pool_ptr);
        }
    }
};

// Get GPU memory info
void printMemoryInfo() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    printf("GPU Memory:\n");
    printf("  Total: %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Free: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used: %.2f GB\n", (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0));
}

// Calculate theoretical peak performance
float getTheoreticalPeakGFLOPS(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    // Peak GFLOPS = cores * clock_rate * 2 (FMA)
    int cores_per_sm = 0;
    
    // Estimate cores based on compute capability
    switch (prop.major) {
        case 7:  // Volta/Turing
            cores_per_sm = (prop.minor == 0) ? 64 : 64;
            break;
        case 8:  // Ampere
            cores_per_sm = 64;
            break;
        default:
            cores_per_sm = 128;  // Assume newer architecture
    }
    
    float peak_gflops = prop.multiProcessorCount * cores_per_sm * 
                       (prop.clockRate / 1e6) * 2.0f;  // 2 for FMA
    
    return peak_gflops;
}

// Check for CUDA errors after kernel launch
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
    
    // Synchronize to catch execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Sync Error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}