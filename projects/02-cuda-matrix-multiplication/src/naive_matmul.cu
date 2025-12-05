#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"

// Forward declarations from utils.cu
class CudaTimer;
void allocateHostMatrices(Matrix& A, Matrix& B, Matrix& C, int N);
void allocateDeviceMatrices(Matrix& d_A, Matrix& d_B, Matrix& d_C, int N);
void transferToDevice(const Matrix h_A, const Matrix h_B, Matrix d_A, Matrix d_B);
void transferFromDevice(Matrix d_C, Matrix h_C);
void freeDeviceMatrices(Matrix d_A, Matrix d_B, Matrix d_C);
void freeHostMatrices(Matrix A, Matrix B, Matrix C);
bool verifyWithCPU(Matrix gpu_result, int N);
void printPerformanceSummary(const char* kernelName, int N, float time_ms);
void printMemoryInfo();

// Naive CUDA kernel - one thread per output element
// Each thread computes one element of C
// Problem: Each thread reads entire row from A and column from B
// This causes massive redundant global memory accesses
__global__ void naiveMatrixMultiplyKernel(
    float* A, float* B, float* C,
    int N)
{
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row >= N || col >= N) return;
    
    // Compute dot product for C[row][col]
    float sum = 0.0f;
    
    // Each thread loads N elements from A and N elements from B
    // This is the bottleneck - O(N) memory accesses per thread
    // Total memory accesses: O(N^3) for N^2 threads
    for (int k = 0; k < N; k++) {
        // Access pattern:
        // - A[row][k]: coalesced across threads in same row
        // - B[k][col]: strided access, poor coalescing
        sum += A[row * N + k] * B[k * N + col];
    }
    
    // Write result
    C[row * N + col] = sum;
}

// Slightly optimized naive kernel with better memory access pattern
__global__ void naiveMatrixMultiplyKernelCoalesced(
    float* A, float* B, float* C,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    
    // Use local variables to potentially keep values in registers
    float a_val, b_val;
    
    for (int k = 0; k < N; k++) {
        a_val = A[row * N + k];
        b_val = B[k * N + col];
        sum += a_val * b_val;
    }
    
    C[row * N + col] = sum;
}

// Wrapper function for kernel launch
void launchNaiveKernel(Matrix d_A, Matrix d_B, Matrix d_C, dim3 grid, dim3 block) {
    naiveMatrixMultiplyKernel<<<grid, block>>>(
        d_A.elements, d_B.elements, d_C.elements, d_A.width);
}

void launchNaiveKernelCoalesced(Matrix d_A, Matrix d_B, Matrix d_C, dim3 grid, dim3 block) {
    naiveMatrixMultiplyKernelCoalesced<<<grid, block>>>(
        d_A.elements, d_B.elements, d_C.elements, d_A.width);
}

#ifndef BENCHMARK_MODE
int main(int argc, char** argv) {
    // Parse command line
    int N = 1024;
    bool useCoalesced = false;
    bool verify = true;
    
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2 && strcmp(argv[2], "coalesced") == 0) {
        useCoalesced = true;
    }
    if (argc > 3 && strcmp(argv[3], "noverify") == 0) {
        verify = false;
    }
    
    printf("Naive CUDA Matrix Multiplication\n");
    printf("=================================\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Kernel version: %s\n", useCoalesced ? "Coalesced" : "Basic");
    
    // Print device info
    printDeviceInfo();
    printMemoryInfo();
    
    // Allocate matrices
    Matrix h_A, h_B, h_C;
    Matrix d_A, d_B, d_C;
    
    printf("Allocating host matrices...\n");
    allocateHostMatrices(h_A, h_B, h_C, N);
    
    printf("Allocating device matrices...\n");
    allocateDeviceMatrices(d_A, d_B, d_C, N);
    
    // Transfer to device
    printf("Transferring data to GPU...\n");
    transferToDevice(h_A, h_B, d_A, d_B);
    
    // Configure kernel launch parameters
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    
    printf("\nKernel configuration:\n");
    printf("  Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("  Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("  Total threads: %d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < 3; i++) {
        if (useCoalesced) {
            naiveMatrixMultiplyKernelCoalesced<<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        } else {
            naiveMatrixMultiplyKernel<<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int iterations = 10;
    printf("Running %d iterations...\n", iterations);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        if (useCoalesced) {
            naiveMatrixMultiplyKernelCoalesced<<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        } else {
            naiveMatrixMultiplyKernel<<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float totalTime;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    float avgTime = totalTime / iterations;
    
    // Print performance
    printPerformanceSummary("Naive CUDA", N, avgTime);
    
    // Analysis of memory access pattern
    printf("\nMemory Access Analysis:\n");
    printf("  Each thread reads: %d floats from A, %d floats from B\n", N, N);
    printf("  Total memory reads per thread: %d floats\n", 2 * N);
    printf("  Global memory accesses: %.2f billion\n", 
           (2.0 * N * N * N) / 1e9);
    
    // Memory bandwidth analysis
    size_t totalBytes = 2L * N * N * N * sizeof(float);  // Read ops
    totalBytes += N * N * sizeof(float);  // Write ops
    float bandwidth = (totalBytes / 1e9) / (avgTime / 1000);
    printf("  Required bandwidth: %.2f GB/s\n", bandwidth);
    
    // Why this is slow
    printf("\nPerformance Bottlenecks:\n");
    printf("  1. Memory bandwidth bound - each element loaded N times\n");
    printf("  2. No data reuse - no shared memory utilization\n");
    printf("  3. Poor memory coalescing for B matrix access\n");
    printf("  4. Low arithmetic intensity: 2 ops per 2 loads\n");
    
    // Transfer result back
    printf("\nTransferring result to host...\n");
    transferFromDevice(d_C, h_C);
    
    // Verification
    if (verify) {
        printf("\nVerifying result...\n");
        if (!verifyWithCPU(h_C, N)) {
            // If CPU result doesn't exist, just print sample values
            printf("Sample results:\n");
            printf("  C[0][0] = %f\n", h_C.elements[0]);
            printf("  C[N/2][N/2] = %f\n", 
                   h_C.elements[IDX2C(N/2, N/2, h_C.stride)]);
            printf("  C[N-1][N-1] = %f\n", 
                   h_C.elements[IDX2C(N-1, N-1, h_C.stride)]);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    freeHostMatrices(h_A, h_B, h_C);
    freeDeviceMatrices(d_A, d_B, d_C);
    
    printf("\nNaive CUDA implementation complete.\n");

    return 0;
}
#endif // BENCHMARK_MODE