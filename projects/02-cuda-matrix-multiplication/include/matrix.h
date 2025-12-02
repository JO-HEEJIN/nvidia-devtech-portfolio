#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// Matrix structure
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Macro for 2D indexing - row-major order
#define IDX2C(i, j, ld) (((i) * (ld)) + (j))

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Helper macro for kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel execution error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Tile size for shared memory - configurable
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// Block size for optimized kernels
#define BLOCK_SIZE 16

// Function declarations

// Matrix allocation and deallocation
inline Matrix allocateMatrix(int width, int height) {
    Matrix mat;
    mat.width = width;
    mat.height = height;
    mat.stride = width;
    size_t size = width * height * sizeof(float);
    mat.elements = (float*)malloc(size);
    return mat;
}

inline Matrix allocateDeviceMatrix(int width, int height) {
    Matrix mat;
    mat.width = width;
    mat.height = height;
    mat.stride = width;
    size_t size = width * height * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&mat.elements, size));
    return mat;
}

inline void freeMatrix(Matrix mat) {
    free(mat.elements);
}

inline void freeDeviceMatrix(Matrix mat) {
    CUDA_CHECK(cudaFree(mat.elements));
}

// Matrix initialization
inline void initializeMatrix(Matrix mat, float value = 0.0f) {
    for (int i = 0; i < mat.height; i++) {
        for (int j = 0; j < mat.width; j++) {
            mat.elements[IDX2C(i, j, mat.stride)] = value;
        }
    }
}

inline void randomizeMatrix(Matrix mat, float min = 0.0f, float max = 1.0f) {
    for (int i = 0; i < mat.height; i++) {
        for (int j = 0; j < mat.width; j++) {
            float random = ((float)rand() / RAND_MAX);
            mat.elements[IDX2C(i, j, mat.stride)] = min + random * (max - min);
        }
    }
}

// Matrix copy operations
inline void copyMatrixHostToDevice(Matrix h_mat, Matrix d_mat) {
    size_t size = h_mat.width * h_mat.height * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_mat.elements, h_mat.elements, size, cudaMemcpyHostToDevice));
}

inline void copyMatrixDeviceToHost(Matrix d_mat, Matrix h_mat) {
    size_t size = h_mat.width * h_mat.height * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_mat.elements, d_mat.elements, size, cudaMemcpyDeviceToHost));
}

// Matrix verification
inline bool verifyResult(Matrix A, Matrix B, float tolerance = 1e-5f) {
    if (A.width != B.width || A.height != B.height) {
        printf("Matrix dimensions don't match: (%d,%d) vs (%d,%d)\n",
               A.height, A.width, B.height, B.width);
        return false;
    }
    
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < A.height; i++) {
        for (int j = 0; j < A.width; j++) {
            int idx = IDX2C(i, j, A.stride);
            float diff = fabs(A.elements[idx] - B.elements[idx]);
            if (diff > tolerance) {
                if (errors < 10) {  // Print first 10 errors
                    printf("Mismatch at (%d,%d): %f vs %f (diff: %e)\n",
                           i, j, A.elements[idx], B.elements[idx], diff);
                }
                errors++;
                max_error = fmax(max_error, diff);
            }
        }
    }
    
    if (errors > 0) {
        printf("Verification FAILED: %d errors, max error: %e\n", errors, max_error);
        return false;
    }
    
    printf("Verification PASSED\n");
    return true;
}

// Performance metrics
inline float calculateGFLOPS(int N, float time_ms) {
    // Matrix multiplication performs 2N^3 operations
    double operations = 2.0 * N * N * N;
    double time_seconds = time_ms / 1000.0;
    return (operations / time_seconds) / 1e9;
}

inline float calculateBandwidth(size_t bytes, float time_ms) {
    // Returns bandwidth in GB/s
    double time_seconds = time_ms / 1000.0;
    return (bytes / time_seconds) / 1e9;
}

// Print matrix (for debugging small matrices)
inline void printMatrix(const char* name, Matrix mat, int max_size = 10) {
    printf("%s (%d x %d):\n", name, mat.height, mat.width);
    int rows = (mat.height < max_size) ? mat.height : max_size;
    int cols = (mat.width < max_size) ? mat.width : max_size;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", mat.elements[IDX2C(i, j, mat.stride)]);
        }
        if (cols < mat.width) printf("...");
        printf("\n");
    }
    if (rows < mat.height) printf("...\n");
    printf("\n");
}

// Device query helper
inline void printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        exit(1);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("Device Information:\n");
    printf("  Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Shared Memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth: %.2f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("\n");
}

#endif // MATRIX_H