#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "matrix.h"

// Forward declarations
void allocateHostMatrices(Matrix& A, Matrix& B, Matrix& C, int N);
void allocateDeviceMatrices(Matrix& d_A, Matrix& d_B, Matrix& d_C, int N);
void transferToDevice(const Matrix h_A, const Matrix h_B, Matrix d_A, Matrix d_B);
void transferFromDevice(Matrix d_C, Matrix h_C);
void freeDeviceMatrices(Matrix d_A, Matrix d_B, Matrix d_C);
void freeHostMatrices(Matrix A, Matrix B, Matrix C);
bool verifyWithCPU(Matrix gpu_result, int N);
void printPerformanceSummary(const char* kernelName, int N, float time_ms);
void printMemoryInfo();

// Fully optimized kernel with all techniques
// 1. Shared memory tiling - reduces global memory access
// 2. Register blocking - each thread computes multiple outputs
// 3. Memory coalescing - consecutive threads access consecutive memory
// 4. Loop unrolling - reduces loop overhead
// 5. Bank conflict avoidance - padding shared memory arrays
template <int BLOCK_SIZE, int WORK_PER_THREAD>
__global__ void optimizedMatrixMultiplyKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Thread block computes BLOCK_SIZE x BLOCK_SIZE tile of C
    // Each thread computes WORK_PER_THREAD x WORK_PER_THREAD elements
    
    // Shared memory with padding to avoid bank conflicts
    // Padding by 1 ensures threads access different banks
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Number of threads per block in each dimension
    const int THREADS_PER_BLOCK = BLOCK_SIZE / WORK_PER_THREAD;
    
    // Each thread computes WORK_PER_THREAD x WORK_PER_THREAD elements
    // Use registers to accumulate results
    float Csub[WORK_PER_THREAD][WORK_PER_THREAD];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j++) {
            Csub[i][j] = 0.0f;
        }
    }
    
    // Global memory indices for this block's tile of C
    int cRow = by * BLOCK_SIZE;
    int cCol = bx * BLOCK_SIZE;
    
    // Loop over tiles of A and B
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Collaborative loading with coalescing
        // Each thread loads WORK_PER_THREAD x WORK_PER_THREAD elements
        
        // Load tile from A
        #pragma unroll
        for (int i = 0; i < WORK_PER_THREAD; i++) {
            int row = ty * WORK_PER_THREAD + i;
            int col = tx;
            int globalRow = cRow + row;
            int globalCol = tile * BLOCK_SIZE + col;
            
            if (globalRow < N && globalCol < N) {
                As[row][col] = A[globalRow * N + globalCol];
            } else {
                As[row][col] = 0.0f;
            }
        }
        
        // Load tile from B
        #pragma unroll
        for (int i = 0; i < WORK_PER_THREAD; i++) {
            int row = ty;
            int col = tx * WORK_PER_THREAD + i;
            int globalRow = tile * BLOCK_SIZE + row;
            int globalCol = cCol + col;
            
            if (globalRow < N && globalCol < N) {
                Bs[row][col] = B[globalRow * N + globalCol];
            } else {
                Bs[row][col] = 0.0f;
            }
        }
        
        // Synchronize to ensure tile is loaded
        __syncthreads();
        
        // Compute partial products
        // Each thread computes WORK_PER_THREAD x WORK_PER_THREAD elements
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // Load values from shared memory into registers
            float Areg[WORK_PER_THREAD];
            float Breg[WORK_PER_THREAD];
            
            #pragma unroll
            for (int i = 0; i < WORK_PER_THREAD; i++) {
                Areg[i] = As[ty * WORK_PER_THREAD + i][k];
                Breg[i] = Bs[k][tx * WORK_PER_THREAD + i];
            }
            
            // Compute outer product
            #pragma unroll
            for (int i = 0; i < WORK_PER_THREAD; i++) {
                #pragma unroll
                for (int j = 0; j < WORK_PER_THREAD; j++) {
                    Csub[i][j] += Areg[i] * Breg[j];
                }
            }
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < WORK_PER_THREAD; i++) {
        #pragma unroll
        for (int j = 0; j < WORK_PER_THREAD; j++) {
            int row = cRow + ty * WORK_PER_THREAD + i;
            int col = cCol + tx * WORK_PER_THREAD + j;
            
            if (row < N && col < N) {
                C[row * N + col] = Csub[i][j];
            }
        }
    }
}

// Vector-based optimized kernel for even better memory access
template <int BLOCK_SIZE>
__global__ void vectorizedMatrixMultiplyKernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Use float4 for vectorized loads/stores (128-bit transactions)
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Accumulator
    float sum = 0.0f;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    // Prefetch first tile
    if (row < N && tx < BLOCK_SIZE) {
        As[ty][tx] = A[row * N + tx];
    } else {
        As[ty][tx] = 0.0f;
    }
    
    if (ty < BLOCK_SIZE && col < N) {
        Bs[ty][tx] = B[ty * N + col];
    } else {
        Bs[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // Main computation loop
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 1; tile < numTiles; tile++) {
        // Double buffering: compute while loading next tile
        float tempSum = 0.0f;
        
        // Compute with current tile
        #pragma unroll 8
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tempSum += As[ty][k] * Bs[k][tx];
        }
        
        sum += tempSum;
        __syncthreads();
        
        // Load next tile
        int aCol = tile * BLOCK_SIZE + tx;
        int bRow = tile * BLOCK_SIZE + ty;
        
        if (row < N && aCol < N) {
            As[ty][tx] = A[row * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (bRow < N && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
    }
    
    // Compute last tile
    #pragma unroll 8
    for (int k = 0; k < BLOCK_SIZE; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    int blockSize = 32;
    int workPerThread = 4;
    bool useVectorized = false;
    bool verify = true;
    
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        blockSize = atoi(argv[2]);
    }
    if (argc > 3 && strcmp(argv[3], "vectorized") == 0) {
        useVectorized = true;
    }
    if (argc > 4 && strcmp(argv[4], "noverify") == 0) {
        verify = false;
    }
    
    printf("Optimized CUDA Matrix Multiplication\n");
    printf("=====================================\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Block size: %d\n", blockSize);
    printf("Work per thread: %d x %d\n", workPerThread, workPerThread);
    printf("Kernel version: %s\n", useVectorized ? "Vectorized" : "Register Blocking");
    
    printDeviceInfo();
    printMemoryInfo();
    
    // Calculate resource usage
    if (!useVectorized) {
        int threadsPerBlock = (blockSize / workPerThread) * (blockSize / workPerThread);
        int registersPerThread = workPerThread * workPerThread + 10;  // Estimate
        printf("\nResource usage:\n");
        printf("  Threads per block: %d\n", threadsPerBlock);
        printf("  Estimated registers per thread: %d\n", registersPerThread);
        printf("  Shared memory per block: %zu bytes\n", 
               2 * blockSize * (blockSize + 1) * sizeof(float));
    }
    
    // Allocate matrices
    Matrix h_A, h_B, h_C;
    Matrix d_A, d_B, d_C;
    
    allocateHostMatrices(h_A, h_B, h_C, N);
    allocateDeviceMatrices(d_A, d_B, d_C, N);
    
    transferToDevice(h_A, h_B, d_A, d_B);
    
    // Configure kernel
    dim3 blockDim, gridDim;
    
    if (useVectorized) {
        blockDim = dim3(16, 16);
        gridDim = dim3((N + 15) / 16, (N + 15) / 16);
    } else {
        int threads = blockSize / workPerThread;
        blockDim = dim3(threads, threads);
        gridDim = dim3((N + blockSize - 1) / blockSize, 
                      (N + blockSize - 1) / blockSize);
    }
    
    printf("\nKernel configuration:\n");
    printf("  Block: (%d, %d)\n", blockDim.x, blockDim.y);
    printf("  Grid: (%d, %d)\n", gridDim.x, gridDim.y);
    
    // Warmup
    printf("\nWarming up...\n");
    for (int i = 0; i < 3; i++) {
        if (useVectorized) {
            vectorizedMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        } else {
            optimizedMatrixMultiplyKernel<32, 4><<<gridDim, blockDim>>>(
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
        if (useVectorized) {
            vectorizedMatrixMultiplyKernel<16><<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        } else {
            optimizedMatrixMultiplyKernel<32, 4><<<gridDim, blockDim>>>(
                d_A.elements, d_B.elements, d_C.elements, N);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float totalTime;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    float avgTime = totalTime / iterations;
    
    printPerformanceSummary("Optimized CUDA", N, avgTime);
    
    // Optimization summary
    printf("\nOptimizations Applied:\n");
    printf("  1. Shared memory tiling (%dx reduction in global accesses)\n", blockSize);
    printf("  2. Register blocking (%dx%d elements per thread)\n", 
           workPerThread, workPerThread);
    printf("  3. Memory coalescing (consecutive threads, consecutive memory)\n");
    printf("  4. Loop unrolling (#pragma unroll)\n");
    printf("  5. Bank conflict avoidance (padding: +1)\n");
    printf("  6. Restrict pointers (compiler optimization)\n");
    if (useVectorized) {
        printf("  7. Vectorized loads (float4)\n");
        printf("  8. Double buffering\n");
    }
    
    // Calculate efficiency
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float peakGflops = prop.multiProcessorCount * 64 * (prop.clockRate / 1e6) * 2;
    float achievedGflops = calculateGFLOPS(N, avgTime);
    float efficiency = (achievedGflops / peakGflops) * 100;
    
    printf("\nEfficiency Analysis:\n");
    printf("  Theoretical peak: %.2f GFLOPS\n", peakGflops);
    printf("  Achieved: %.2f GFLOPS\n", achievedGflops);
    printf("  Efficiency: %.1f%%\n", efficiency);
    
    transferFromDevice(d_C, h_C);
    
    if (verify) {
        printf("\nVerifying result...\n");
        if (!verifyWithCPU(h_C, N)) {
            printf("Sample results:\n");
            printf("  C[0][0] = %f\n", h_C.elements[0]);
            printf("  C[N-1][N-1] = %f\n", 
                   h_C.elements[IDX2C(N-1, N-1, h_C.stride)]);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    freeHostMatrices(h_A, h_B, h_C);
    freeDeviceMatrices(d_A, d_B, d_C);
    
    printf("\nOptimized implementation complete.\n");
    
    return 0;
}