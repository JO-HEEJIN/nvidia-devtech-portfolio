# CUDA Image Processing Makefile

# Compiler settings
NVCC = nvcc
CXX = g++

# Architecture settings (adjust based on your GPU)
GPU_ARCH = -arch=sm_75 -arch=sm_80 -arch=sm_86

# Directories
SRCDIR = src
KERNELDIR = src/kernels
INCDIR = include
BUILDDIR = build
DEMODIR = demo

# OpenCV settings
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Compiler flags
NVCC_FLAGS = -std=c++14 $(GPU_ARCH) -O3 -lineinfo
NVCC_DEBUG_FLAGS = -std=c++14 $(GPU_ARCH) -O0 -g -G -lineinfo
CXX_FLAGS = -std=c++14 -O3 -Wall -Wextra
CXX_DEBUG_FLAGS = -std=c++14 -O0 -g -Wall -Wextra

# Include paths
INCLUDES = -I$(INCDIR) -I/usr/local/cuda/include $(OPENCV_CFLAGS)

# Library paths
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda $(OPENCV_LIBS) -lm

# Source files
KERNEL_SOURCES = $(wildcard $(KERNELDIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)
DEMO_SOURCES = $(wildcard $(DEMODIR)/*.cpp)

# Object files
KERNEL_OBJECTS = $(KERNEL_SOURCES:$(KERNELDIR)/%.cu=$(BUILDDIR)/%.o)
CPP_OBJECTS = $(CPP_SOURCES:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)
DEMO_OBJECTS = $(DEMO_SOURCES:$(DEMODIR)/%.cpp=$(BUILDDIR)/demo_%.o)

# Target executables
MAIN_TARGET = cuda_image_processor
BENCHMARK_TARGET = benchmark
DEMO_TARGET = demo

# Default target
all: dirs release

# Create build directories
dirs:
	@mkdir -p $(BUILDDIR)

# Release build
release: NVCC_FLAGS += -O3
release: CXX_FLAGS += -O3 -DNDEBUG
release: $(MAIN_TARGET) $(BENCHMARK_TARGET) $(DEMO_TARGET)

# Debug build
debug: NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
debug: CXX_FLAGS = $(CXX_DEBUG_FLAGS)
debug: $(MAIN_TARGET) $(BENCHMARK_TARGET) $(DEMO_TARGET)

# Main executable
$(MAIN_TARGET): $(KERNEL_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Benchmark executable
$(BENCHMARK_TARGET): $(KERNEL_OBJECTS) $(BUILDDIR)/benchmark.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Demo executable
$(DEMO_TARGET): $(KERNEL_OBJECTS) $(BUILDDIR)/demo_process_image.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Compile CUDA kernel files
$(BUILDDIR)/%.o: $(KERNELDIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile demo files
$(BUILDDIR)/demo_%.o: $(DEMODIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Special targets for individual kernels (for testing)
test_grayscale: dirs $(BUILDDIR)/grayscale.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

test_blur: dirs $(BUILDDIR)/gaussian_blur.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

test_sobel: dirs $(BUILDDIR)/sobel_edge.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

test_histogram: dirs $(BUILDDIR)/histogram.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

test_resize: dirs $(BUILDDIR)/resize.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

test_convolution: dirs $(BUILDDIR)/convolution.o $(BUILDDIR)/main.o
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# Install target (optional)
install: release
	@echo "Installing executables to /usr/local/bin..."
	@sudo cp $(MAIN_TARGET) /usr/local/bin/
	@sudo cp $(BENCHMARK_TARGET) /usr/local/bin/cuda_image_benchmark
	@sudo cp $(DEMO_TARGET) /usr/local/bin/cuda_image_demo

# Clean build files
clean:
	rm -rf $(BUILDDIR)
	rm -f $(MAIN_TARGET) $(BENCHMARK_TARGET) $(DEMO_TARGET)
	rm -f test_*

# Deep clean (including generated images)
distclean: clean
	rm -f *.png *.jpg *.jpeg *.bmp
	rm -f sample_images/output_*

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version || (echo "CUDA not found! Please install CUDA toolkit." && exit 1)
	@echo "Checking GPU devices..."
	@nvidia-smi || (echo "NVIDIA driver not found!" && exit 1)

# Check OpenCV installation
check-opencv:
	@echo "Checking OpenCV installation..."
	@pkg-config --exists opencv4 || pkg-config --exists opencv || (echo "OpenCV not found!" && exit 1)
	@echo "OpenCV version: $$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv)"

# Check all dependencies
check: check-cuda check-opencv
	@echo "All dependencies found!"

# Run tests (requires sample images)
test: release
	@echo "Running basic functionality tests..."
	@./$(MAIN_TARGET) --help
	@echo "Tests completed!"

# Profile with nsight compute (requires sudo/proper setup)
profile: release
	@echo "Profiling with Nsight Compute..."
	@ncu --target-processes all --force-overwrite -o profile ./$(BENCHMARK_TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build all targets (release mode)"
	@echo "  release     - Build optimized release version"
	@echo "  debug       - Build debug version with symbols"
	@echo "  clean       - Remove build files"
	@echo "  distclean   - Remove all generated files"
	@echo "  check       - Check CUDA and OpenCV installation"
	@echo "  test        - Run basic functionality tests"
	@echo "  install     - Install executables to system"
	@echo "  profile     - Profile with Nsight Compute"
	@echo ""
	@echo "Individual kernel tests:"
	@echo "  test_grayscale    - Test grayscale conversion"
	@echo "  test_blur         - Test Gaussian blur"
	@echo "  test_sobel        - Test Sobel edge detection"
	@echo "  test_histogram    - Test histogram calculation"
	@echo "  test_resize       - Test image resize"
	@echo "  test_convolution  - Test convolution"

.PHONY: all release debug clean distclean check check-cuda check-opencv test install profile help dirs