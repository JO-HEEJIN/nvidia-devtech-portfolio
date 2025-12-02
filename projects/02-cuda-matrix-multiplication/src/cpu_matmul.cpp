#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h>
#include "matrix.h"

// CPU matrix multiplication - reference implementation
void cpuMatrixMultiply(const Matrix A, const Matrix B, Matrix C) {
    // Triple nested loop - O(n^3) complexity
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

// Optimized CPU version with OpenMP parallelization
void cpuMatrixMultiplyOMP(const Matrix A, const Matrix B, Matrix C) {
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

// Tiled CPU version for better cache usage
void cpuMatrixMultiplyTiled(const Matrix A, const Matrix B, Matrix C, int tileSize = 32) {
    // Initialize result matrix to zero
    memset(C.elements, 0, C.width * C.height * sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < A.height; ii += tileSize) {
        for (int jj = 0; jj < B.width; jj += tileSize) {
            for (int kk = 0; kk < A.width; kk += tileSize) {
                // Process tile
                int iMax = std::min(ii + tileSize, A.height);
                int jMax = std::min(jj + tileSize, B.width);
                int kMax = std::min(kk + tileSize, A.width);
                
                for (int i = ii; i < iMax; i++) {
                    for (int j = jj; j < jMax; j++) {
                        float sum = C.elements[IDX2C(i, j, C.stride)];
                        for (int k = kk; k < kMax; k++) {
                            sum += A.elements[IDX2C(i, k, A.stride)] * 
                                   B.elements[IDX2C(k, j, B.stride)];
                        }
                        C.elements[IDX2C(i, j, C.stride)] = sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int N = 1024;  // Default matrix size
    bool useOpenMP = true;
    bool useTiling = false;
    
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2 && strcmp(argv[2], "notiling") == 0) {
        useTiling = false;
    }
    if (argc > 3 && strcmp(argv[3], "noomp") == 0) {
        useOpenMP = false;
    }
    
    printf("CPU Matrix Multiplication\n");
    printf("=========================\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("OpenMP: %s\n", useOpenMP ? "Enabled" : "Disabled");
    printf("Tiling: %s\n", useTiling ? "Enabled" : "Disabled");
    
    // Set number of OpenMP threads
    if (useOpenMP) {
        int numThreads = omp_get_max_threads();
        printf("OpenMP threads: %d\n", numThreads);
        omp_set_num_threads(numThreads);
    }
    
    // Allocate matrices
    Matrix A = allocateMatrix(N, N);
    Matrix B = allocateMatrix(N, N);
    Matrix C = allocateMatrix(N, N);
    
    // Initialize matrices
    srand(42);  // Fixed seed for reproducibility
    randomizeMatrix(A, 0.0f, 1.0f);
    randomizeMatrix(B, 0.0f, 1.0f);
    initializeMatrix(C, 0.0f);
    
    printf("\nMatrix initialization complete\n");
    
    // Warm-up run
    if (useTiling) {
        cpuMatrixMultiplyTiled(A, B, C);
    } else if (useOpenMP) {
        cpuMatrixMultiplyOMP(A, B, C);
    } else {
        cpuMatrixMultiply(A, B, C);
    }
    
    // Benchmark
    const int numIterations = (N <= 512) ? 5 : 1;
    double totalTime = 0.0;
    
    printf("\nRunning %d iterations...\n", numIterations);
    
    for (int iter = 0; iter < numIterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (useTiling) {
            cpuMatrixMultiplyTiled(A, B, C);
        } else if (useOpenMP) {
            cpuMatrixMultiplyOMP(A, B, C);
        } else {
            cpuMatrixMultiply(A, B, C);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        totalTime += elapsed.count();
        
        if (iter == 0) {
            printf("First iteration: %.2f ms\n", elapsed.count());
        }
    }
    
    double avgTime = totalTime / numIterations;
    
    // Calculate performance metrics
    float gflops = calculateGFLOPS(N, avgTime);
    size_t memoryOps = 2L * N * N * N * sizeof(float);  // Read A and B for each output
    float bandwidth = calculateBandwidth(memoryOps, avgTime);
    
    // Print results
    printf("\nPerformance Results:\n");
    printf("-------------------\n");
    printf("Average time: %.2f ms\n", avgTime);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    // Print sample of result matrix for verification
    if (N <= 8) {
        printMatrix("Result C", C);
    } else {
        printf("\nSample result (C[0][0]): %f\n", C.elements[0]);
        printf("Sample result (C[N-1][N-1]): %f\n", 
               C.elements[IDX2C(N-1, N-1, C.stride)]);
    }
    
    // Save result for verification with GPU versions
    FILE* fp = fopen("cpu_result.bin", "wb");
    if (fp) {
        fwrite(C.elements, sizeof(float), N * N, fp);
        fclose(fp);
        printf("\nCPU result saved to cpu_result.bin for verification\n");
    }
    
    // Free memory
    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(C);
    
    return 0;
}