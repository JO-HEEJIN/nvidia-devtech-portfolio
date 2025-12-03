#include "cuda_utils.cuh"
#include "image_io.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <map>

// CUDA kernel function declarations
extern "C" {
    // Grayscale kernels
    void launch_grayscale_simple(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, cudaStream_t stream);
    void launch_grayscale_vectorized(const unsigned char* d_input, unsigned char* d_output,
                                   int width, int height, cudaStream_t stream);
    void launch_grayscale_coalesced(const unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, cudaStream_t stream);
    void launch_grayscale_auto(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream);
    
    // Histogram kernels
    void launch_histogram_atomic_global(const unsigned char* d_input, unsigned int* d_histogram,
                                       int width, int height, cudaStream_t stream);
    void launch_histogram_shared_privatization(const unsigned char* d_input, unsigned int* d_histogram,
                                              int width, int height, cudaStream_t stream);
    void launch_histogram_reduction(const unsigned char* d_input, unsigned int* d_histogram,
                                   int width, int height, cudaStream_t stream);
    void launch_histogram_warp_optimized(const unsigned char* d_input, unsigned int* d_histogram,
                                        int width, int height, cudaStream_t stream);
    void launch_histogram_auto(const unsigned char* d_input, unsigned int* d_histogram,
                              int width, int height, cudaStream_t stream);
    
    // Gaussian blur kernels
    void launch_gaussian_blur_naive(const unsigned char* d_input, unsigned char* d_output,
                                   int width, int height, int radius, float sigma, cudaStream_t stream);
    void launch_gaussian_blur_shared(const unsigned char* d_input, unsigned char* d_output,
                                    int width, int height, int radius, float sigma, cudaStream_t stream);
    void launch_gaussian_blur_separable(const unsigned char* d_input, unsigned char* d_output,
                                       int width, int height, int radius, float sigma,
                                       unsigned char* d_temp, cudaStream_t stream);
    void launch_gaussian_blur_texture(const unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, int radius, float sigma, cudaStream_t stream);
    void launch_gaussian_blur_auto(const unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, int radius, float sigma, cudaStream_t stream);
    
    // Sobel edge detection kernels
    void launch_sobel_edge_naive(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, cudaStream_t stream);
    void launch_sobel_edge_shared(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, cudaStream_t stream);
    void launch_sobel_edge_threshold(const unsigned char* d_input, unsigned char* d_output,
                                    int width, int height, float threshold, cudaStream_t stream);
    void launch_sobel_edge_auto(const unsigned char* d_input, unsigned char* d_output,
                               int width, int height, cudaStream_t stream);
    
    // Resize kernels
    void launch_resize_nearest_neighbor(const unsigned char* d_input, unsigned char* d_output,
                                       int src_width, int src_height,
                                       int dst_width, int dst_height, cudaStream_t stream);
    void launch_resize_bilinear(const unsigned char* d_input, unsigned char* d_output,
                               int src_width, int src_height,
                               int dst_width, int dst_height, cudaStream_t stream);
    void launch_resize_bicubic(const unsigned char* d_input, unsigned char* d_output,
                              int src_width, int src_height,
                              int dst_width, int dst_height, cudaStream_t stream);
    void launch_resize_auto(const unsigned char* d_input, unsigned char* d_output,
                           int src_width, int src_height,
                           int dst_width, int dst_height, cudaStream_t stream);
    
    // Convolution kernels
    void setup_convolution_kernel(const float* kernel, int kernel_size);
    void launch_convolution_naive(const unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, int kernel_size, cudaStream_t stream);
    void launch_convolution_shared(const unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, int kernel_size, cudaStream_t stream);
    void launch_sharpen_filter(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, cudaStream_t stream);
    void launch_edge_detection_filter(const unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, cudaStream_t stream);
}

struct BenchmarkResult {
    std::string operation;
    std::string variant;
    double time_ms;
    double speedup;
    size_t memory_usage_mb;
};

class ImageBenchmark {
private:
    std::vector<BenchmarkResult> results;
    cv::Mat test_image;
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned char* d_temp;
    unsigned int* d_histogram;
    int width, height, channels;
    cudaStream_t stream;
    
public:
    ImageBenchmark(const std::string& image_path) {
        // Load test image
        test_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (test_image.empty()) {
            // Create synthetic test image if file doesn't exist
            width = 1920;
            height = 1080;
            test_image = cv::Mat(height, width, CV_8UC1);
            cv::randu(test_image, cv::Scalar::all(0), cv::Scalar::all(255));
            std::cout << "Created synthetic test image (" << width << "x" << height << ")" << std::endl;
        } else {
            width = test_image.cols;
            height = test_image.rows;
            std::cout << "Loaded test image: " << image_path << " (" << width << "x" << height << ")" << std::endl;
        }
        channels = test_image.channels();
        
        // Allocate GPU memory
        size_t image_size = width * height * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_input, image_size));
        CUDA_CHECK(cudaMalloc(&d_output, image_size));
        CUDA_CHECK(cudaMalloc(&d_temp, image_size));
        CUDA_CHECK(cudaMalloc(&d_histogram, 256 * sizeof(unsigned int)));
        
        // Copy test image to GPU
        CUDA_CHECK(cudaMemcpy(d_input, test_image.data, image_size, cudaMemcpyHostToDevice));
        
        // Create CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~ImageBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_temp);
        cudaFree(d_histogram);
        cudaStreamDestroy(stream);
    }
    
    double benchmark_opencv_operation(const std::string& operation, int iterations = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::Mat result;
            
            if (operation == "grayscale") {
                cv::cvtColor(test_image, result, cv::COLOR_GRAY2GRAY);
            } else if (operation == "histogram") {
                std::vector<cv::Mat> bgr_planes = {test_image};
                cv::Mat hist;
                int histSize = 256;
                float range[] = {0, 256};
                const float* histRange = {range};
                cv::calcHist(&bgr_planes, {0}, cv::Mat(), hist, {histSize}, {histRange});
            } else if (operation == "blur") {
                cv::GaussianBlur(test_image, result, cv::Size(15, 15), 2.0);
            } else if (operation == "sobel") {
                cv::Mat grad_x, grad_y, grad;
                cv::Sobel(test_image, grad_x, CV_16S, 1, 0, 3);
                cv::Sobel(test_image, grad_y, CV_16S, 0, 1, 3);
                cv::convertScaleAbs(grad_x, grad_x);
                cv::convertScaleAbs(grad_y, grad_y);
                cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
            } else if (operation == "resize") {
                cv::resize(test_image, result, cv::Size(width/2, height/2), 0, 0, cv::INTER_LINEAR);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        return duration;
    }
    
    template<typename KernelFunc, typename... Args>
    double benchmark_cuda_kernel(KernelFunc kernel, int iterations, Args... args) {
        // Warm-up
        for (int i = 0; i < 5; i++) {
            kernel(args...);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            kernel(args...);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        double duration = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        return duration;
    }
    
    void benchmark_grayscale_operations(int iterations = 100) {
        std::cout << "\n=== Grayscale Conversion Benchmarks ===" << std::endl;
        
        // OpenCV baseline
        double opencv_time = benchmark_opencv_operation("grayscale", iterations);
        std::cout << "OpenCV (CPU): " << std::fixed << std::setprecision(3) << opencv_time << " ms" << std::endl;
        
        // CUDA variants
        auto test_grayscale_variant = [&](auto func, const std::string& name) {
            double cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_output, width, height, stream);
            double speedup = opencv_time / cuda_time;
            results.push_back({"grayscale", name, cuda_time, speedup, (width * height * 2) / (1024 * 1024)});
            std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
        };
        
        test_grayscale_variant(launch_grayscale_simple, "Simple");
        test_grayscale_variant(launch_grayscale_vectorized, "Vectorized");
        test_grayscale_variant(launch_grayscale_coalesced, "Coalesced");
        test_grayscale_variant(launch_grayscale_auto, "Auto-Select");
    }
    
    void benchmark_histogram_operations(int iterations = 100) {
        std::cout << "\n=== Histogram Calculation Benchmarks ===" << std::endl;
        
        // OpenCV baseline
        double opencv_time = benchmark_opencv_operation("histogram", iterations);
        std::cout << "OpenCV (CPU): " << std::fixed << std::setprecision(3) << opencv_time << " ms" << std::endl;
        
        // CUDA variants
        auto test_histogram_variant = [&](auto func, const std::string& name) {
            double cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_histogram, width, height, stream);
            double speedup = opencv_time / cuda_time;
            results.push_back({"histogram", name, cuda_time, speedup, (width * height + 256 * 4) / (1024 * 1024)});
            std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
        };
        
        test_histogram_variant(launch_histogram_atomic_global, "Atomic Global");
        test_histogram_variant(launch_histogram_shared_privatization, "Shared Memory");
        test_histogram_variant(launch_histogram_reduction, "Reduction");
        test_histogram_variant(launch_histogram_warp_optimized, "Warp Optimized");
        test_histogram_variant(launch_histogram_auto, "Auto-Select");
    }
    
    void benchmark_blur_operations(int iterations = 50) {
        std::cout << "\n=== Gaussian Blur Benchmarks ===" << std::endl;
        
        int radius = 5;
        float sigma = 2.0f;
        
        // OpenCV baseline
        double opencv_time = benchmark_opencv_operation("blur", iterations);
        std::cout << "OpenCV (CPU): " << std::fixed << std::setprecision(3) << opencv_time << " ms" << std::endl;
        
        // CUDA variants
        auto test_blur_variant = [&](auto func, const std::string& name, bool needs_temp = false) {
            if (needs_temp) {
                double cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_output, 
                                                       width, height, radius, sigma, d_temp, stream);
                double speedup = opencv_time / cuda_time;
                results.push_back({"blur", name, cuda_time, speedup, (width * height * 3) / (1024 * 1024)});
                std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
            } else {
                double cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_output, 
                                                       width, height, radius, sigma, stream);
                double speedup = opencv_time / cuda_time;
                results.push_back({"blur", name, cuda_time, speedup, (width * height * 2) / (1024 * 1024)});
                std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
            }
        };
        
        test_blur_variant(launch_gaussian_blur_naive, "Naive");
        test_blur_variant(launch_gaussian_blur_shared, "Shared Memory");
        test_blur_variant(launch_gaussian_blur_separable, "Separable", true);
        test_blur_variant(launch_gaussian_blur_texture, "Texture Memory");
        test_blur_variant(launch_gaussian_blur_auto, "Auto-Select");
    }
    
    void benchmark_edge_detection(int iterations = 100) {
        std::cout << "\n=== Edge Detection Benchmarks ===" << std::endl;
        
        // OpenCV baseline
        double opencv_time = benchmark_opencv_operation("sobel", iterations);
        std::cout << "OpenCV (CPU): " << std::fixed << std::setprecision(3) << opencv_time << " ms" << std::endl;
        
        // CUDA variants
        auto test_edge_variant = [&](auto func, const std::string& name, bool has_threshold = false) {
            double cuda_time;
            if (has_threshold) {
                cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_output, width, height, 50.0f, stream);
            } else {
                cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_output, width, height, stream);
            }
            double speedup = opencv_time / cuda_time;
            results.push_back({"sobel", name, cuda_time, speedup, (width * height * 2) / (1024 * 1024)});
            std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
        };
        
        test_edge_variant(launch_sobel_edge_naive, "Naive");
        test_edge_variant(launch_sobel_edge_shared, "Shared Memory");
        test_edge_variant(launch_sobel_edge_threshold, "Thresholded", true);
        test_edge_variant(launch_sobel_edge_auto, "Auto-Select");
    }
    
    void benchmark_resize_operations(int iterations = 100) {
        std::cout << "\n=== Resize Benchmarks ===" << std::endl;
        
        int dst_width = width / 2;
        int dst_height = height / 2;
        
        // Allocate output for resize
        unsigned char* d_resize_output;
        CUDA_CHECK(cudaMalloc(&d_resize_output, dst_width * dst_height * sizeof(unsigned char)));
        
        // OpenCV baseline
        double opencv_time = benchmark_opencv_operation("resize", iterations);
        std::cout << "OpenCV (CPU): " << std::fixed << std::setprecision(3) << opencv_time << " ms" << std::endl;
        
        // CUDA variants
        auto test_resize_variant = [&](auto func, const std::string& name) {
            double cuda_time = benchmark_cuda_kernel(func, iterations, d_input, d_resize_output, 
                                                   width, height, dst_width, dst_height, stream);
            double speedup = opencv_time / cuda_time;
            size_t mem_usage = (width * height + dst_width * dst_height) / (1024 * 1024);
            results.push_back({"resize", name, cuda_time, speedup, mem_usage});
            std::cout << name << ": " << cuda_time << " ms (Speedup: " << speedup << "x)" << std::endl;
        };
        
        test_resize_variant(launch_resize_nearest_neighbor, "Nearest Neighbor");
        test_resize_variant(launch_resize_bilinear, "Bilinear");
        test_resize_variant(launch_resize_bicubic, "Bicubic");
        test_resize_variant(launch_resize_auto, "Auto-Select");
        
        cudaFree(d_resize_output);
    }
    
    void run_all_benchmarks() {
        std::cout << "CUDA Image Processing Performance Benchmark" << std::endl;
        std::cout << "Image size: " << width << "x" << height << " pixels" << std::endl;
        
        // Print GPU information
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        
        benchmark_grayscale_operations();
        benchmark_histogram_operations();
        benchmark_blur_operations();
        benchmark_edge_detection();
        benchmark_resize_operations();
        
        print_summary();
    }
    
    void print_summary() {
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::left << std::setw(15) << "Operation" 
                  << std::setw(20) << "Variant" 
                  << std::setw(12) << "Time (ms)"
                  << std::setw(10) << "Speedup"
                  << std::setw(12) << "Memory (MB)" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(15) << result.operation
                      << std::setw(20) << result.variant
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.time_ms
                      << std::setw(10) << std::fixed << std::setprecision(1) << result.speedup << "x"
                      << std::setw(12) << result.memory_usage_mb << std::endl;
        }
        
        // Find best performing variant for each operation
        std::map<std::string, BenchmarkResult> best_results;
        for (const auto& result : results) {
            if (best_results.find(result.operation) == best_results.end() ||
                result.speedup > best_results[result.operation].speedup) {
                best_results[result.operation] = result;
            }
        }
        
        std::cout << "\n=== Best Performing Variants ===" << std::endl;
        for (const auto& [operation, result] : best_results) {
            std::cout << operation << ": " << result.variant 
                      << " (" << std::fixed << std::setprecision(1) << result.speedup << "x speedup)" << std::endl;
        }
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i <image_path>     Input image path (optional, creates synthetic if not provided)" << std::endl;
    std::cout << "  -op <operation>     Benchmark specific operation (grayscale, histogram, blur, sobel, resize, all)" << std::endl;
    std::cout << "  -iterations <n>     Number of iterations for each benchmark (default: 100)" << std::endl;
    std::cout << "  -help               Show this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "";
    std::string operation = "all";
    int iterations = 100;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-i" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (std::string(argv[i]) == "-op" && i + 1 < argc) {
            operation = argv[++i];
        } else if (std::string(argv[i]) == "-iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "-help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    try {
        ImageBenchmark benchmark(image_path);
        
        if (operation == "all") {
            benchmark.run_all_benchmarks();
        } else if (operation == "grayscale") {
            benchmark.benchmark_grayscale_operations(iterations);
        } else if (operation == "histogram") {
            benchmark.benchmark_histogram_operations(iterations);
        } else if (operation == "blur") {
            benchmark.benchmark_blur_operations(iterations);
        } else if (operation == "sobel") {
            benchmark.benchmark_edge_detection(iterations);
        } else if (operation == "resize") {
            benchmark.benchmark_resize_operations(iterations);
        } else {
            std::cerr << "Unknown operation: " << operation << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}