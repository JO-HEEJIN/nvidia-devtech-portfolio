#include "cuda_utils.cuh"
#include "image_io.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <map>
#include <chrono>

// CUDA kernel function declarations
extern "C" {
    // Grayscale
    void launch_grayscale_auto(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream);
    
    // Histogram
    void launch_histogram_auto(const unsigned char* d_input, unsigned int* d_histogram,
                              int width, int height, cudaStream_t stream);
    void compute_histogram_stats(const unsigned int* h_histogram, int total_pixels,
                                float* mean, float* std_dev, int* min_val, int* max_val);
    
    // Gaussian blur
    void launch_gaussian_blur_auto(const unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, int radius, float sigma, cudaStream_t stream);
    
    // Sobel edge detection
    void launch_sobel_edge_auto(const unsigned char* d_input, unsigned char* d_output,
                               int width, int height, cudaStream_t stream);
    void launch_sobel_edge_threshold(const unsigned char* d_input, unsigned char* d_output,
                                    int width, int height, float threshold, cudaStream_t stream);
    
    // Resize
    void launch_resize_auto(const unsigned char* d_input, unsigned char* d_output,
                           int src_width, int src_height,
                           int dst_width, int dst_height, cudaStream_t stream);
    
    // Convolution filters
    void launch_sharpen_filter(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, cudaStream_t stream);
    void launch_edge_detection_filter(const unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, cudaStream_t stream);
    void launch_emboss_filter(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream);
}

class ImageProcessor {
private:
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned char* d_temp;
    unsigned int* d_histogram;
    cudaStream_t stream;
    
public:
    ImageProcessor() {
        d_input = nullptr;
        d_output = nullptr;
        d_temp = nullptr;
        d_histogram = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~ImageProcessor() {
        cleanup();
        cudaStreamDestroy(stream);
    }
    
    void allocate_memory(int width, int height, int channels = 1) {
        cleanup();
        
        size_t image_size = width * height * channels * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_input, image_size));
        CUDA_CHECK(cudaMalloc(&d_output, image_size));
        CUDA_CHECK(cudaMalloc(&d_temp, image_size));
        CUDA_CHECK(cudaMalloc(&d_histogram, 256 * sizeof(unsigned int)));
    }
    
    void cleanup() {
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        if (d_temp) { cudaFree(d_temp); d_temp = nullptr; }
        if (d_histogram) { cudaFree(d_histogram); d_histogram = nullptr; }
    }
    
    void upload_image(const cv::Mat& image) {
        size_t image_size = image.total() * image.elemSize();
        CUDA_CHECK(cudaMemcpy(d_input, image.data, image_size, cudaMemcpyHostToDevice));
    }
    
    void download_image(cv::Mat& image) {
        size_t image_size = image.total() * image.elemSize();
        CUDA_CHECK(cudaMemcpy(image.data, d_output, image_size, cudaMemcpyDeviceToHost));
    }
    
    double process_grayscale(int width, int height) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_grayscale_auto(d_input, d_output, width, height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_histogram(int width, int height, bool print_stats = false) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_histogram_auto(d_input, d_histogram, width, height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        if (print_stats) {
            // Download histogram and compute statistics
            unsigned int h_histogram[256];
            CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(unsigned int), 
                                 cudaMemcpyDeviceToHost));
            
            float mean, std_dev;
            int min_val, max_val;
            compute_histogram_stats(h_histogram, width * height, &mean, &std_dev, &min_val, &max_val);
            
            std::cout << "Histogram Statistics:" << std::endl;
            std::cout << "  Mean: " << mean << std::endl;
            std::cout << "  Std Dev: " << std_dev << std::endl;
            std::cout << "  Min Value: " << min_val << std::endl;
            std::cout << "  Max Value: " << max_val << std::endl;
        }
        
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_blur(int width, int height, int radius, float sigma) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_gaussian_blur_auto(d_input, d_output, width, height, radius, sigma, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_sobel(int width, int height, float threshold = -1.0f) {
        auto start = std::chrono::high_resolution_clock::now();
        if (threshold > 0.0f) {
            launch_sobel_edge_threshold(d_input, d_output, width, height, threshold, stream);
        } else {
            launch_sobel_edge_auto(d_input, d_output, width, height, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_resize(int src_width, int src_height, int dst_width, int dst_height) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_resize_auto(d_input, d_output, src_width, src_height, dst_width, dst_height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_sharpen(int width, int height) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_sharpen_filter(d_input, d_output, width, height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_edge_filter(int width, int height) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_edge_detection_filter(d_input, d_output, width, height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    double process_emboss(int width, int height) {
        auto start = std::chrono::high_resolution_clock::now();
        launch_emboss_filter(d_input, d_output, width, height, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

void print_usage(const char* program_name) {
    std::cout << "CUDA Image Processor - High-performance GPU image processing" << std::endl;
    std::cout << "\nUsage: " << program_name << " -i <input> -o <output> -op <operation> [options]" << std::endl;
    std::cout << "\nRequired arguments:" << std::endl;
    std::cout << "  -i <path>           Input image path" << std::endl;
    std::cout << "  -o <path>           Output image path" << std::endl;
    std::cout << "  -op <operation>     Operation to perform" << std::endl;
    std::cout << "\nSupported operations:" << std::endl;
    std::cout << "  grayscale           Convert to grayscale" << std::endl;
    std::cout << "  histogram           Compute histogram (no output image)" << std::endl;
    std::cout << "  blur                Gaussian blur" << std::endl;
    std::cout << "  sobel               Sobel edge detection" << std::endl;
    std::cout << "  resize              Resize image" << std::endl;
    std::cout << "  sharpen             Sharpen image" << std::endl;
    std::cout << "  edge                Edge detection filter" << std::endl;
    std::cout << "  emboss              Emboss filter" << std::endl;
    std::cout << "\nOperation-specific options:" << std::endl;
    std::cout << "  -radius <n>         Blur radius (default: 5)" << std::endl;
    std::cout << "  -sigma <f>          Blur sigma (default: 2.0)" << std::endl;
    std::cout << "  -threshold <f>      Edge detection threshold (default: auto)" << std::endl;
    std::cout << "  -scale <f>          Resize scale factor (default: 0.5)" << std::endl;
    std::cout << "  -width <n>          Resize target width" << std::endl;
    std::cout << "  -height <n>         Resize target height" << std::endl;
    std::cout << "\nGeneral options:" << std::endl;
    std::cout << "  -verbose            Show processing time and details" << std::endl;
    std::cout << "  -stats              Show histogram statistics (for histogram operation)" << std::endl;
    std::cout << "  -help               Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " -i input.jpg -o output.jpg -op grayscale" << std::endl;
    std::cout << "  " << program_name << " -i input.jpg -o blurred.jpg -op blur -radius 7 -sigma 3.0" << std::endl;
    std::cout << "  " << program_name << " -i input.jpg -o edges.jpg -op sobel -threshold 100" << std::endl;
    std::cout << "  " << program_name << " -i input.jpg -o small.jpg -op resize -scale 0.25" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_path, output_path, operation;
    int radius = 5;
    float sigma = 2.0f;
    float threshold = -1.0f;
    float scale = 0.5f;
    int target_width = -1, target_height = -1;
    bool verbose = false;
    bool show_stats = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-op" && i + 1 < argc) {
            operation = argv[++i];
        } else if (arg == "-radius" && i + 1 < argc) {
            radius = std::stoi(argv[++i]);
        } else if (arg == "-sigma" && i + 1 < argc) {
            sigma = std::stof(argv[++i]);
        } else if (arg == "-threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg == "-scale" && i + 1 < argc) {
            scale = std::stof(argv[++i]);
        } else if (arg == "-width" && i + 1 < argc) {
            target_width = std::stoi(argv[++i]);
        } else if (arg == "-height" && i + 1 < argc) {
            target_height = std::stoi(argv[++i]);
        } else if (arg == "-verbose") {
            verbose = true;
        } else if (arg == "-stats") {
            show_stats = true;
        } else if (arg == "-help" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (input_path.empty() || operation.empty()) {
        std::cerr << "Error: Input path and operation are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (output_path.empty() && operation != "histogram") {
        std::cerr << "Error: Output path is required for this operation" << std::endl;
        return 1;
    }
    
    try {
        // Load input image
        cv::Mat input_image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
        if (input_image.empty()) {
            std::cerr << "Error: Could not load input image: " << input_path << std::endl;
            return 1;
        }
        
        if (verbose) {
            std::cout << "Loaded image: " << input_path << " (" << input_image.cols 
                      << "x" << input_image.rows << ")" << std::endl;
            
            // Print GPU information
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            std::cout << "Using GPU: " << prop.name << std::endl;
        }
        
        // Initialize processor
        ImageProcessor processor;
        processor.allocate_memory(input_image.cols, input_image.rows);
        processor.upload_image(input_image);
        
        // Process image based on operation
        double processing_time = 0.0;
        cv::Mat output_image;
        
        if (operation == "grayscale") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_grayscale(input_image.cols, input_image.rows);
            processor.download_image(output_image);
            
        } else if (operation == "histogram") {
            processing_time = processor.process_histogram(input_image.cols, input_image.rows, show_stats);
            
        } else if (operation == "blur") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_blur(input_image.cols, input_image.rows, radius, sigma);
            processor.download_image(output_image);
            
        } else if (operation == "sobel") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_sobel(input_image.cols, input_image.rows, threshold);
            processor.download_image(output_image);
            
        } else if (operation == "resize") {
            int dst_width, dst_height;
            if (target_width > 0 && target_height > 0) {
                dst_width = target_width;
                dst_height = target_height;
            } else {
                dst_width = static_cast<int>(input_image.cols * scale);
                dst_height = static_cast<int>(input_image.rows * scale);
            }
            
            output_image = cv::Mat::zeros(dst_height, dst_width, CV_8UC1);
            processing_time = processor.process_resize(input_image.cols, input_image.rows, 
                                                     dst_width, dst_height);
            processor.download_image(output_image);
            
        } else if (operation == "sharpen") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_sharpen(input_image.cols, input_image.rows);
            processor.download_image(output_image);
            
        } else if (operation == "edge") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_edge_filter(input_image.cols, input_image.rows);
            processor.download_image(output_image);
            
        } else if (operation == "emboss") {
            output_image = cv::Mat::zeros(input_image.size(), CV_8UC1);
            processing_time = processor.process_emboss(input_image.cols, input_image.rows);
            processor.download_image(output_image);
            
        } else {
            std::cerr << "Error: Unknown operation: " << operation << std::endl;
            return 1;
        }
        
        // Save output image if needed
        if (!output_image.empty() && !output_path.empty()) {
            if (!cv::imwrite(output_path, output_image)) {
                std::cerr << "Error: Could not save output image: " << output_path << std::endl;
                return 1;
            }
            
            if (verbose) {
                std::cout << "Saved output: " << output_path << " (" << output_image.cols 
                          << "x" << output_image.rows << ")" << std::endl;
            }
        }
        
        if (verbose) {
            std::cout << "Processing time: " << std::fixed << std::setprecision(3) 
                      << processing_time << " ms" << std::endl;
        }
        
        std::cout << "Operation '" << operation << "' completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}