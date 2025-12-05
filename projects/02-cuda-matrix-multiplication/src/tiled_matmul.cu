#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"

// Forward declarations from utils.cu
void allocateHostMatrices(Matrix& A, Matrix& B, Matrix& C, int N);
void allocateDeviceMatrices(Matrix& d_A, Matrix& d_B, Matrix& d_C, int N);
void transferToDevice(const Matrix h_A, const Matrix h_B, Matrix d_A, Matrix d_B);
void transferFromDevice(Matrix d_C, Matrix h_C);
void freeDeviceMatrices(Matrix d_A, Matrix d_B, Matrix d_C);
void freeHostMatrices(Matrix A, Matrix B, Matrix C);
bool verifyWithCPU(Matrix gpu_result, int N);
void printPerformanceSummary(const char* kernelName, int N, float time_ms);
void printMemoryInfo();

// Tiled matrix multiplication kernel using shared memory
// Key optimization: Load tiles of A and B into shared memory
// This reduces global memory accesses by factor of TILE_SIZE
template <int TILE_WIDTH>
__global__ void tiledMatrixMultiplyKernel(
    float* A, float* B, float* C, int N)
{
    // Allocate shared memory for tiles
    // These tiles are shared by all threads in the block
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column for this thread's output element
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Loop over tiles of A and B required to compute C element
    // Number of tiles = ceil(N / TILE_WIDTH)
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        // Collaborative loading: Each thread loads one element
        // of each tile into shared memory
        
        // Load element from A
        int aRow = row;
        int aCol = tile * TILE_WIDTH + tx;
        if (aRow < N && aCol < N) {
            As[ty][tx] = A[aRow * N + aCol];
        } else {
            As[ty][tx] = 0.0f;  // Pad with zeros for boundary
        }
        
        // Load element from B
        int bRow = tile * TILE_WIDTH + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            Bs[ty][tx] = B[bRow * N + bCol];
        } else {
            Bs[ty][tx] = 0.0f;  // Pad with zeros for boundary
        }
        
        // Synchronize to ensure all threads have loaded their data
        // This is critical - without it, threads might use uninitialized data
        __syncthreads();
        
        // Compute partial dot product using shared memory
        // Each thread computes one element of the tile product
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        // Ensures all threads are done using current tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Advanced tiled kernel with boundary condition optimization
template <int TILE_WIDTH>
__global__ void tiledMatrixMultiplyKernelOptimized(
    float* __restrict__ A, 
    float* __restrict__ B, 
    float* __restrict__ C, 
    int N)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Use register to accumulate result
    float sum = 0.0f;
    
    // Calculate starting points
    int aBegin = N * by * TILE_WIDTH;
    int aEnd = aBegin + N - 1;
    int aStep = TILE_WIDTH;
    
    int bBegin = bx * TILE_WIDTH;
    int bStep = TILE_WIDTH * N;
    
    // Loop over tiles
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load tiles with coalesced access
        // Check boundaries only once per tile
        if (a + tx < N * N && (a / N) + ty < N) {
            As[ty][tx] = A[a + N * ty + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (b + N * ty + tx < N * N && (b % N) + tx < N) {
            Bs[ty][tx] = B[b + N * ty + tx];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute tile product
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    int c = N * by * TILE_WIDTH + bx * TILE_WIDTH;
    if (c + N * ty + tx < N * N && by * TILE_WIDTH + ty < N && bx * TILE_WIDTH + tx < N) {
        C[c + N * ty + tx] = sum;
    }
}

#ifndef BENCHMARK_MODE
int main(int argc, char** argv) {
    // Parse command line
    int N = 1024;
    int tileSize = 16;  // Default tile size
    bool useOptimized = false;
    bool verify = true;
    
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        tileSize = atoi(argv[2]);
    }
    if (argc > 3 && strcmp(argv[3], "optimized") == 0) {
        useOptimized = true;
    }
    if (argc > 4 && strcmp(argv[4], "noverify") == 0) {
        verify = false;
    }
    
    printf("Tiled CUDA Matrix Multiplication\n");
    printf("=================================\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size: %d x %d\n", tileSize, tileSize);
    printf("Kernel version: %s\n", useOptimized ? "Optimized" : "Basic");
    
    // Print device info
    printDeviceInfo();
    printMemoryInfo();
    
    // Calculate shared memory usage
    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(float);
    printf("\nShared memory per block: %zu bytes\n", sharedMemSize);
    
    // Check if tile size is valid
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (sharedMemSize > prop.sharedMemPerBlock) {
        printf("Error: Tile size too large for device shared memory!\n");
        printf("Maximum shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        return 1;
    }
    
    // Allocate matrices
    Matrix h_A, h_B, h_C;
    Matrix d_A, d_B, d_C;
    
    printf("\nAllocating matrices...\n");
    allocateHostMatrices(h_A, h_B, h_C, N);
    allocateDeviceMatrices(d_A, d_B, d_C, N);
    
    // Transfer to device
    printf("Transferring data to GPU...\n");
    transferToDevice(h_A, h_B, d_A, d_B);
    
    // Configure kernel launch
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, 
                 (N + tileSize - 1) / tileSize);
    
    printf("\nKernel configuration:\n");
    printf("  Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("  Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("  Total blocks: %d\n", gridDim.x * gridDim.y);
    
    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < 3; i++) {
        if (tileSize == 16) {
            if (useOptimized) {
                tiledMatrixMultiplyKernelOptimized<16><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            } else {
                tiledMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            }
        } else if (tileSize == 32) {
            if (useOptimized) {
                tiledMatrixMultiplyKernelOptimized<32><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            } else {
                tiledMatrixMultiplyKernel<32><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            }
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
        if (tileSize == 16) {
            if (useOptimized) {
                tiledMatrixMultiplyKernelOptimized<16><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            } else {
                tiledMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            }
        } else if (tileSize == 32) {
            if (useOptimized) {
                tiledMatrixMultiplyKernelOptimized<32><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            } else {
                tiledMatrixMultiplyKernel<32><<<gridDim, blockDim>>>(
                    d_A.elements, d_B.elements, d_C.elements, N);
            }
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float totalTime;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    float avgTime = totalTime / iterations;
    
    // Print performance
    printPerformanceSummary("Tiled CUDA", N, avgTime);
    
    // Analysis of shared memory optimization
    printf("\nShared Memory Optimization Analysis:\n");
    printf("  Tile size: %d x %d\n", tileSize, tileSize);
    printf("  Global memory accesses reduced by: %dx\n", tileSize);
    printf("  Shared memory loads per block: %d\n", 2 * tileSize * tileSize);
    printf("  Arithmetic operations per block: %d\n", tileSize * tileSize * tileSize);
    printf("  Arithmetic intensity: %.2f ops/byte\n", 
           (float)(2 * tileSize) / (2 * sizeof(float)));
    
    // Why this is faster
    printf("\nPerformance Improvements:\n");
    printf("  1. Data reuse - each tile loaded once, used %d times\n", tileSize);
    printf("  2. Shared memory latency ~%dx faster than global\n", 100);
    printf("  3. Better cache utilization - tiles fit in L1\n");
    printf("  4. Coalesced memory access for both A and B\n");
    printf("  5. Higher arithmetic intensity: %.1f vs 1.0\n", (float)tileSize / 2);
    
    // Transfer result back
    printf("\nTransferring result to host...\n");
    transferFromDevice(d_C, h_C);
    
    // Verification
    if (verify) {
        printf("\nVerifying result...\n");
        if (!verifyWithCPU(h_C, N)) {
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
    
    printf("\nTiled CUDA implementation complete.\n");

    return 0;
}
#endif // BENCHMARK_MODE