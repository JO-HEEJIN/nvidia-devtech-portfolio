#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <iomanip>
#include "matrix.h"

// Forward declarations
void allocateHostMatrices(Matrix& A, Matrix& B, Matrix& C, int N);
void allocateDeviceMatrices(Matrix& d_A, Matrix& d_B, Matrix& d_C, int N);
void transferToDevice(const Matrix h_A, const Matrix h_B, Matrix d_A, Matrix d_B);
void transferFromDevice(Matrix d_C, Matrix h_C);
void freeDeviceMatrices(Matrix d_A, Matrix d_B, Matrix d_C);
void freeHostMatrices(Matrix A, Matrix B, Matrix C);
void printPerformanceSummary(const char* kernelName, int N, float time_ms);

// Import kernels from other files
extern __global__ void naiveMatrixMultiplyKernel(float* A, float* B, float* C, int N);

template <int TILE_WIDTH>
extern __global__ void tiledMatrixMultiplyKernel(float* A, float* B, float* C, int N);

template <int BLK_SIZE, int WORK_PER_THREAD>
extern __global__ void optimizedMatrixMultiplyKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N);

// CPU baseline function
void cpuMatrixMultiply(const Matrix A, const Matrix B, Matrix C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < B.width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.width; k++) {
                sum += A.elements[IDX2C(i, k, A.stride)] * 
                       B.elements[IDX2C(k, j, B.stride)];
            }
            C.elements[IDX2C(i, j, C.stride)] = sum;
        }
    }
}

// Benchmark result structure
struct BenchmarkResult {
    const char* name;
    float time_ms;
    float gflops;
    float bandwidth_gb;
    float speedup;
};

// Run CPU benchmark
BenchmarkResult benchmarkCPU(Matrix A, Matrix B, Matrix C, int N, int iterations = 1) {
    BenchmarkResult result;
    result.name = "CPU (OpenMP)";
    
    // Warmup
    cpuMatrixMultiply(A, B, C);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cpuMatrixMultiply(A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    result.time_ms = elapsed.count() / iterations;
    result.gflops = calculateGFLOPS(N, result.time_ms);
    
    size_t memoryOps = 2L * N * N * N * sizeof(float) + N * N * sizeof(float);
    result.bandwidth_gb = calculateBandwidth(memoryOps, result.time_ms);
    result.speedup = 1.0f;  // Baseline
    
    return result;
}

// Run naive CUDA benchmark
BenchmarkResult benchmarkNaiveCUDA(Matrix d_A, Matrix d_B, Matrix d_C, int N, int iterations = 10) {
    BenchmarkResult result;
    result.name = "Naive CUDA";
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        naiveMatrixMultiplyKernel<<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        naiveMatrixMultiplyKernel<<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    result.gflops = calculateGFLOPS(N, result.time_ms);
    
    size_t memoryOps = 2L * N * N * N * sizeof(float) + N * N * sizeof(float);
    result.bandwidth_gb = calculateBandwidth(memoryOps, result.time_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return result;
}

// Run tiled CUDA benchmark
BenchmarkResult benchmarkTiledCUDA(Matrix d_A, Matrix d_B, Matrix d_C, int N, int iterations = 10) {
    BenchmarkResult result;
    result.name = "Tiled CUDA (16x16)";
    
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        tiledMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        tiledMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    result.gflops = calculateGFLOPS(N, result.time_ms);
    
    size_t memoryOps = (2L * N * N * N / TILE_SIZE + N * N) * sizeof(float);
    result.bandwidth_gb = calculateBandwidth(memoryOps, result.time_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return result;
}

// Run optimized CUDA benchmark
BenchmarkResult benchmarkOptimizedCUDA(Matrix d_A, Matrix d_B, Matrix d_C, int N, int iterations = 10) {
    BenchmarkResult result;
    result.name = "Optimized CUDA";
    
    const int BLK_SIZE = 32;
    const int WORK_PER_THREAD = 4;
    int threads = BLK_SIZE / WORK_PER_THREAD;
    
    dim3 blockDim(threads, threads);
    dim3 gridDim((N + BLK_SIZE - 1) / BLK_SIZE,
                 (N + BLK_SIZE - 1) / BLK_SIZE);
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        optimizedMatrixMultiplyKernel<32, 4><<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        optimizedMatrixMultiplyKernel<32, 4><<<gridDim, blockDim>>>(
            d_A.elements, d_B.elements, d_C.elements, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    result.gflops = calculateGFLOPS(N, result.time_ms);
    
    size_t memoryOps = (2L * N * N * N / BLK_SIZE + N * N) * sizeof(float);
    result.bandwidth_gb = calculateBandwidth(memoryOps, result.time_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return result;
}

// Run cuBLAS benchmark
BenchmarkResult benchmarkCuBLAS(Matrix d_A, Matrix d_B, Matrix d_C, int N, int iterations = 10) {
    BenchmarkResult result;
    result.name = "cuBLAS";
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_B.elements, N,
                    d_A.elements, N,
                    &beta, d_C.elements, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_B.elements, N,
                    d_A.elements, N,
                    &beta, d_C.elements, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    result.gflops = calculateGFLOPS(N, result.time_ms);
    
    size_t memoryOps = (N * N + N * N + N * N) * sizeof(float);
    result.bandwidth_gb = calculateBandwidth(memoryOps, result.time_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(handle);
    
    return result;
}

// Print results table
void printResultsTable(std::vector<BenchmarkResult>& results, float cpuTime) {
    printf("\n");
    printf("========================================================================\n");
    printf("                          BENCHMARK RESULTS                            \n");
    printf("========================================================================\n");
    printf("%-20s %12s %12s %12s %12s\n", 
           "Implementation", "Time (ms)", "GFLOPS", "GB/s", "Speedup");
    printf("------------------------------------------------------------------------\n");
    
    for (auto& result : results) {
        result.speedup = cpuTime / result.time_ms;
        printf("%-20s %12.3f %12.2f %12.2f %12.1fx\n",
               result.name, result.time_ms, result.gflops, 
               result.bandwidth_gb, result.speedup);
    }
    
    printf("========================================================================\n");
}

int main(int argc, char** argv) {
    // Parse command line
    int startSize = 256;
    int endSize = 2048;
    bool quickMode = false;
    
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printf("Usage: %s [size] [quick]\n", argv[0]);
            printf("  size: Single matrix size to test\n");
            printf("  quick: Run quick benchmark (smaller sizes)\n");
            return 0;
        }
        startSize = endSize = atoi(argv[1]);
    }
    if (argc > 2 && strcmp(argv[2], "quick") == 0) {
        quickMode = true;
        if (argc == 2) {  // No size specified
            startSize = 256;
            endSize = 1024;
        }
    }
    
    printf("CUDA Matrix Multiplication Benchmark Suite\n");
    printf("===========================================\n");
    
    // Print device info
    printDeviceInfo();
    printMemoryInfo();
    
    // Test sizes
    std::vector<int> sizes;
    if (startSize == endSize) {
        sizes.push_back(startSize);
    } else {
        for (int n = startSize; n <= endSize; n *= 2) {
            sizes.push_back(n);
        }
    }
    
    // Run benchmarks for each size
    for (int N : sizes) {
        printf("\n\nMatrix Size: %d x %d\n", N, N);
        printf("Total Operations: %.2f GFLOP\n", 2.0 * N * N * N / 1e9);
        printf("Memory Footprint: %.2f MB\n", 3.0 * N * N * sizeof(float) / (1024 * 1024));
        
        // Allocate matrices
        Matrix h_A, h_B, h_C, h_C_verify;
        Matrix d_A, d_B, d_C;
        
        allocateHostMatrices(h_A, h_B, h_C, N);
        h_C_verify = allocateMatrix(N, N);
        allocateDeviceMatrices(d_A, d_B, d_C, N);
        
        transferToDevice(h_A, h_B, d_A, d_B);
        
        std::vector<BenchmarkResult> results;
        
        // CPU benchmark (skip for large sizes in quick mode)
        float cpuTime = 0;
        if (N <= 1024 || !quickMode) {
            printf("\nRunning CPU benchmark...\n");
            auto cpuResult = benchmarkCPU(h_A, h_B, h_C_verify, N, N <= 512 ? 3 : 1);
            cpuTime = cpuResult.time_ms;
            results.push_back(cpuResult);
            
            // Save CPU result for verification
            FILE* fp = fopen("cpu_result.bin", "wb");
            if (fp) {
                fwrite(h_C_verify.elements, sizeof(float), N * N, fp);
                fclose(fp);
            }
        } else {
            cpuTime = 1000000;  // Large number for skipped CPU
            printf("\nSkipping CPU benchmark (too slow for N=%d)\n", N);
        }
        
        // GPU benchmarks
        printf("Running GPU benchmarks...\n");
        
        // Naive CUDA
        results.push_back(benchmarkNaiveCUDA(d_A, d_B, d_C, N));
        
        // Tiled CUDA
        results.push_back(benchmarkTiledCUDA(d_A, d_B, d_C, N));
        
        // Optimized CUDA
        results.push_back(benchmarkOptimizedCUDA(d_A, d_B, d_C, N));
        
        // cuBLAS
        results.push_back(benchmarkCuBLAS(d_A, d_B, d_C, N));
        
        // Print results
        printResultsTable(results, cpuTime);
        
        // Performance analysis
        printf("\nPerformance Analysis:\n");
        printf("---------------------\n");
        
        // Find best performer
        float bestTime = results[0].time_ms;
        const char* bestName = results[0].name;
        for (auto& result : results) {
            if (result.time_ms < bestTime) {
                bestTime = result.time_ms;
                bestName = result.name;
            }
        }
        printf("Best performer: %s\n", bestName);
        
        // Calculate improvements
        if (results.size() >= 4) {
            float naiveTime = results[1].time_ms;
            float tiledTime = results[2].time_ms;
            float optTime = results[3].time_ms;
            
            printf("Tiled vs Naive: %.1fx speedup\n", naiveTime / tiledTime);
            printf("Optimized vs Tiled: %.1fx speedup\n", tiledTime / optTime);
            printf("Optimized vs Naive: %.1fx speedup\n", naiveTime / optTime);
        }
        
        // Cleanup
        freeMatrix(h_C_verify);
        freeHostMatrices(h_A, h_B, h_C);
        freeDeviceMatrices(d_A, d_B, d_C);
    }
    
    printf("\n\nBenchmark complete!\n");
    
    return 0;
}