# CUDA Matrix Multiplication Makefile
# Supports multiple GPU architectures and build modes

# Compiler and flags
NVCC = nvcc
CXX = g++
CUDA_PATH ?= /usr/local/cuda

# Architecture flags - support multiple GPUs
# SM_70: V100, SM_75: T4, SM_80: A100, SM_86: RTX 3090
GPU_ARCH = -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86

# Compilation flags
NVCC_FLAGS = $(GPU_ARCH) -O3 -std=c++14 -lineinfo
CXX_FLAGS = -O3 -std=c++14 -Xcompiler -fopenmp
DEBUG_FLAGS = -g -G -O0 -DDEBUG

# Include and library paths
INCLUDES = -I$(CUDA_PATH)/include -I./include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Source and object files
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Create directories
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

# Targets (GPU only - cpu_matmul requires OpenMP which may not be available)
TARGETS = $(BIN_DIR)/naive_matmul \
          $(BIN_DIR)/tiled_matmul \
          $(BIN_DIR)/optimized_matmul \
          $(BIN_DIR)/benchmark

all: $(TARGETS)

# CPU baseline (optional - requires OpenMP)
cpu: $(BIN_DIR)/cpu_matmul

# CPU baseline
$(BIN_DIR)/cpu_matmul: $(SRC_DIR)/cpu_matmul.cpp $(SRC_DIR)/utils.cu
	$(NVCC) $(CXX_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# Naive CUDA implementation
$(BIN_DIR)/naive_matmul: $(SRC_DIR)/naive_matmul.cu $(SRC_DIR)/utils.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# Tiled implementation
$(BIN_DIR)/tiled_matmul: $(SRC_DIR)/tiled_matmul.cu $(SRC_DIR)/utils.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# Optimized implementation
$(BIN_DIR)/optimized_matmul: $(SRC_DIR)/optimized_matmul.cu $(SRC_DIR)/utils.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# Benchmark suite
$(BIN_DIR)/benchmark: $(SRC_DIR)/benchmark.cu $(SRC_DIR)/utils.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $^ -o $@ $(LIBS)

# Debug builds
debug: NVCC_FLAGS = $(GPU_ARCH) $(DEBUG_FLAGS) -std=c++14
debug: CXX_FLAGS = $(DEBUG_FLAGS) -std=c++14 -Xcompiler -fopenmp
debug: all

# Individual debug targets
debug-naive: NVCC_FLAGS = $(GPU_ARCH) $(DEBUG_FLAGS) -std=c++14
debug-naive: $(BIN_DIR)/naive_matmul

debug-tiled: NVCC_FLAGS = $(GPU_ARCH) $(DEBUG_FLAGS) -std=c++14
debug-tiled: $(BIN_DIR)/tiled_matmul

debug-optimized: NVCC_FLAGS = $(GPU_ARCH) $(DEBUG_FLAGS) -std=c++14
debug-optimized: $(BIN_DIR)/optimized_matmul

# Profile builds (with lineinfo for nsys/ncu)
profile: NVCC_FLAGS = $(GPU_ARCH) -O3 -std=c++14 -lineinfo -Xcompiler -rdynamic
profile: all

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -f *.o *.exe *.exp *.lib
	rm -f reports/*.nsys-rep reports/*.ncu-rep

# Run tests with default size
test: all
	@echo "Running CPU baseline (N=1024)..."
	$(BIN_DIR)/cpu_matmul 1024
	@echo "\nRunning Naive CUDA (N=1024)..."
	$(BIN_DIR)/naive_matmul 1024
	@echo "\nRunning Tiled CUDA (N=1024)..."
	$(BIN_DIR)/tiled_matmul 1024
	@echo "\nRunning Optimized CUDA (N=1024)..."
	$(BIN_DIR)/optimized_matmul 1024

# Run full benchmark
benchmark: $(BIN_DIR)/benchmark
	@echo "Running full benchmark suite..."
	$(BIN_DIR)/benchmark

# Help
help:
	@echo "CUDA Matrix Multiplication Makefile"
	@echo "===================================="
	@echo "Targets:"
	@echo "  all              - Build all implementations"
	@echo "  debug            - Build with debug symbols"
	@echo "  profile          - Build for profiling"
	@echo "  test             - Run quick test (N=1024)"
	@echo "  benchmark        - Run full benchmark"
	@echo "  clean            - Remove build artifacts"
	@echo ""
	@echo "Individual targets:"
	@echo "  make naive_matmul     - Build naive CUDA version"
	@echo "  make tiled_matmul     - Build tiled version"
	@echo "  make optimized_matmul - Build optimized version"
	@echo ""
	@echo "Debug targets:"
	@echo "  make debug-naive      - Debug build of naive version"
	@echo "  make debug-tiled      - Debug build of tiled version"
	@echo ""
	@echo "GPU Architectures supported:"
	@echo "  SM_70 (V100), SM_75 (T4), SM_80 (A100), SM_86 (RTX 3090)"

.PHONY: all debug profile test benchmark clean help