#include "cuda_utils.cuh"
#include "image_io.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>

// CUDA kernel function declarations
extern "C" {
    // All available kernels from the main application
    void launch_grayscale_auto(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream);
    void launch_histogram_auto(const unsigned char* d_input, unsigned int* d_histogram,
                              int width, int height, cudaStream_t stream);
    void launch_gaussian_blur_auto(const unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, int radius, float sigma, cudaStream_t stream);
    void launch_sobel_edge_auto(const unsigned char* d_input, unsigned char* d_output,
                               int width, int height, cudaStream_t stream);
    void launch_sobel_edge_threshold(const unsigned char* d_input, unsigned char* d_output,
                                    int width, int height, float threshold, cudaStream_t stream);
    void launch_resize_auto(const unsigned char* d_input, unsigned char* d_output,
                           int src_width, int src_height, int dst_width, int dst_height, cudaStream_t stream);
    void launch_sharpen_filter(const unsigned char* d_input, unsigned char* d_output,
                              int width, int height, cudaStream_t stream);
    void launch_emboss_filter(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, cudaStream_t stream);
    void launch_edge_detection_filter(const unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, cudaStream_t stream);
}

class InteractiveImageProcessor {
private:
    cv::Mat original_image;
    cv::Mat current_image;
    cv::Mat display_image;
    std::vector<std::string> processing_pipeline;
    std::map<std::string, std::string> filter_params;
    
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned char* d_temp;
    unsigned int* d_histogram;
    cudaStream_t stream;
    
    // Default parameters
    int blur_radius = 3;
    float blur_sigma = 1.5f;
    float sobel_threshold = 50.0f;
    float resize_scale = 1.0f;
    
public:
    InteractiveImageProcessor() {
        d_input = nullptr;
        d_output = nullptr;
        d_temp = nullptr;
        d_histogram = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~InteractiveImageProcessor() {
        cleanup();
        cudaStreamDestroy(stream);
    }
    
    void cleanup() {
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        if (d_temp) { cudaFree(d_temp); d_temp = nullptr; }
        if (d_histogram) { cudaFree(d_histogram); d_histogram = nullptr; }
    }
    
    bool load_image(const std::string& path) {
        original_image = cv::imread(path, cv::IMREAD_COLOR);
        if (original_image.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return false;
        }
        
        // Convert to grayscale for processing
        cv::cvtColor(original_image, current_image, cv::COLOR_BGR2GRAY);
        current_image.copyTo(display_image);
        
        // Allocate GPU memory
        allocate_gpu_memory();
        
        std::cout << "Loaded image: " << path << " (" << current_image.cols 
                  << "x" << current_image.rows << ")" << std::endl;
        return true;
    }
    
    void allocate_gpu_memory() {
        cleanup();
        
        int width = current_image.cols;
        int height = current_image.rows;
        size_t image_size = width * height * sizeof(unsigned char);
        
        CUDA_CHECK(cudaMalloc(&d_input, image_size));
        CUDA_CHECK(cudaMalloc(&d_output, image_size));
        CUDA_CHECK(cudaMalloc(&d_temp, image_size));
        CUDA_CHECK(cudaMalloc(&d_histogram, 256 * sizeof(unsigned int)));
    }
    
    void upload_current_image() {
        size_t image_size = current_image.total() * current_image.elemSize();
        CUDA_CHECK(cudaMemcpy(d_input, current_image.data, image_size, cudaMemcpyHostToDevice));
    }
    
    void download_processed_image() {
        size_t image_size = current_image.total() * current_image.elemSize();
        CUDA_CHECK(cudaMemcpy(current_image.data, d_output, image_size, cudaMemcpyDeviceToHost));
        current_image.copyTo(display_image);
    }
    
    void apply_filter(const std::string& filter_name) {
        auto start = std::chrono::high_resolution_clock::now();
        
        upload_current_image();
        
        int width = current_image.cols;
        int height = current_image.rows;
        
        if (filter_name == "grayscale") {
            // Already grayscale, so this is a no-op for demo
            current_image.copyTo(display_image);
            
        } else if (filter_name == "blur") {
            launch_gaussian_blur_auto(d_input, d_output, width, height, 
                                     blur_radius, blur_sigma, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            download_processed_image();
            
        } else if (filter_name == "sobel") {
            if (sobel_threshold > 0) {
                launch_sobel_edge_threshold(d_input, d_output, width, height, 
                                          sobel_threshold, stream);
            } else {
                launch_sobel_edge_auto(d_input, d_output, width, height, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            download_processed_image();
            
        } else if (filter_name == "sharpen") {
            launch_sharpen_filter(d_input, d_output, width, height, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            download_processed_image();
            
        } else if (filter_name == "emboss") {
            launch_emboss_filter(d_input, d_output, width, height, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            download_processed_image();
            
        } else if (filter_name == "edge") {
            launch_edge_detection_filter(d_input, d_output, width, height, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            download_processed_image();
            
        } else if (filter_name == "histogram") {
            launch_histogram_auto(d_input, d_histogram, width, height, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Download and display histogram
            show_histogram();
            
        } else {
            std::cout << "Unknown filter: " << filter_name << std::endl;
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        processing_pipeline.push_back(filter_name);
        std::cout << "Applied " << filter_name << " filter (" << std::fixed 
                  << std::setprecision(2) << duration << " ms)" << std::endl;
    }
    
    void show_histogram() {
        // Download histogram data
        unsigned int h_histogram[256];
        CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(unsigned int), 
                             cudaMemcpyDeviceToHost));
        
        // Create histogram visualization
        int hist_w = 512, hist_h = 400;
        int bin_w = cvRound(static_cast<double>(hist_w) / 256);
        cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Normalize histogram
        unsigned int max_val = *std::max_element(h_histogram, h_histogram + 256);
        
        // Draw histogram
        for (int i = 1; i < 256; i++) {
            cv::line(hist_image,
                    cv::Point(bin_w * (i - 1), hist_h - cvRound(h_histogram[i - 1] * hist_h / max_val)),
                    cv::Point(bin_w * i, hist_h - cvRound(h_histogram[i] * hist_h / max_val)),
                    cv::Scalar(255, 255, 255), 2, 8, 0);
        }
        
        cv::imshow("Histogram", hist_image);
    }
    
    void reset_image() {
        cv::cvtColor(original_image, current_image, cv::COLOR_BGR2GRAY);
        current_image.copyTo(display_image);
        processing_pipeline.clear();
        std::cout << "Reset to original image" << std::endl;
    }
    
    void save_image(const std::string& path) {
        if (cv::imwrite(path, display_image)) {
            std::cout << "Saved processed image to: " << path << std::endl;
        } else {
            std::cerr << "Failed to save image: " << path << std::endl;
        }
    }
    
    void show_image(const std::string& window_name = "CUDA Image Processor") {
        if (!display_image.empty()) {
            cv::imshow(window_name, display_image);
        }
    }
    
    void set_blur_params(int radius, float sigma) {
        blur_radius = radius;
        blur_sigma = sigma;
        std::cout << "Set blur parameters: radius=" << radius << ", sigma=" << sigma << std::endl;
    }
    
    void set_sobel_threshold(float threshold) {
        sobel_threshold = threshold;
        std::cout << "Set Sobel threshold: " << threshold << std::endl;
    }
    
    void print_pipeline() {
        std::cout << "\nProcessing pipeline: ";
        for (size_t i = 0; i < processing_pipeline.size(); i++) {
            std::cout << processing_pipeline[i];
            if (i < processing_pipeline.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
    }
    
    void print_help() {
        std::cout << "\n=== Interactive CUDA Image Processor ===" << std::endl;
        std::cout << "Available commands:" << std::endl;
        std::cout << "  blur         - Apply Gaussian blur" << std::endl;
        std::cout << "  sobel        - Apply Sobel edge detection" << std::endl;
        std::cout << "  sharpen      - Apply sharpening filter" << std::endl;
        std::cout << "  emboss       - Apply emboss filter" << std::endl;
        std::cout << "  edge         - Apply edge detection filter" << std::endl;
        std::cout << "  histogram    - Show histogram" << std::endl;
        std::cout << "  reset        - Reset to original image" << std::endl;
        std::cout << "  save <path>  - Save current image" << std::endl;
        std::cout << "  pipeline     - Show processing pipeline" << std::endl;
        std::cout << "  params       - Show current parameters" << std::endl;
        std::cout << "  set-blur <radius> <sigma> - Set blur parameters" << std::endl;
        std::cout << "  set-sobel <threshold>     - Set Sobel threshold" << std::endl;
        std::cout << "  help         - Show this help" << std::endl;
        std::cout << "  quit         - Exit program" << std::endl;
        std::cout << "\nPress any key in the image window to refresh display" << std::endl;
    }
    
    void print_params() {
        std::cout << "\nCurrent parameters:" << std::endl;
        std::cout << "  Blur radius: " << blur_radius << std::endl;
        std::cout << "  Blur sigma: " << blur_sigma << std::endl;
        std::cout << "  Sobel threshold: " << sobel_threshold << std::endl;
    }
    
    void run_interactive_mode() {
        print_help();
        
        std::string command;
        while (true) {
            show_image();
            cv::waitKey(1); // Allow window updates
            
            std::cout << "\nEnter command: ";
            std::getline(std::cin, command);
            
            if (command.empty()) continue;
            
            std::vector<std::string> tokens;
            std::stringstream ss(command);
            std::string token;
            while (ss >> token) {
                tokens.push_back(token);
            }
            
            if (tokens[0] == "quit" || tokens[0] == "exit") {
                break;
            } else if (tokens[0] == "help") {
                print_help();
            } else if (tokens[0] == "reset") {
                reset_image();
            } else if (tokens[0] == "pipeline") {
                print_pipeline();
            } else if (tokens[0] == "params") {
                print_params();
            } else if (tokens[0] == "save" && tokens.size() > 1) {
                save_image(tokens[1]);
            } else if (tokens[0] == "set-blur" && tokens.size() > 2) {
                try {
                    int radius = std::stoi(tokens[1]);
                    float sigma = std::stof(tokens[2]);
                    set_blur_params(radius, sigma);
                } catch (const std::exception& e) {
                    std::cout << "Invalid parameters for set-blur" << std::endl;
                }
            } else if (tokens[0] == "set-sobel" && tokens.size() > 1) {
                try {
                    float threshold = std::stof(tokens[1]);
                    set_sobel_threshold(threshold);
                } catch (const std::exception& e) {
                    std::cout << "Invalid threshold for set-sobel" << std::endl;
                }
            } else if (tokens[0] == "blur" || tokens[0] == "sobel" || 
                      tokens[0] == "sharpen" || tokens[0] == "emboss" || 
                      tokens[0] == "edge" || tokens[0] == "histogram") {
                apply_filter(tokens[0]);
            } else {
                std::cout << "Unknown command. Type 'help' for available commands." << std::endl;
            }
        }
    }
    
    void run_batch_mode(const std::vector<std::string>& filters) {
        std::cout << "Running batch processing with filters: ";
        for (size_t i = 0; i < filters.size(); i++) {
            std::cout << filters[i];
            if (i < filters.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        
        for (const auto& filter : filters) {
            apply_filter(filter);
            show_image("Batch Processing");
            cv::waitKey(500); // Show each step
        }
        
        print_pipeline();
        std::cout << "Batch processing completed. Press any key to exit." << std::endl;
        cv::waitKey(0);
    }
};

void print_usage(const char* program_name) {
    std::cout << "Interactive CUDA Image Processor Demo" << std::endl;
    std::cout << "\nUsage: " << program_name << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -i <image_path>     Input image path (required)" << std::endl;
    std::cout << "  -pipeline <filters> Comma-separated list of filters for batch processing" << std::endl;
    std::cout << "  -blur-radius <n>    Default blur radius (default: 3)" << std::endl;
    std::cout << "  -blur-sigma <f>     Default blur sigma (default: 1.5)" << std::endl;
    std::cout << "  -sobel-threshold <f> Default Sobel threshold (default: 50.0)" << std::endl;
    std::cout << "  -help               Show this help message" << std::endl;
    std::cout << "\nAvailable filters for pipeline:" << std::endl;
    std::cout << "  blur, sobel, sharpen, emboss, edge, histogram" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << " -i sample.jpg" << std::endl;
    std::cout << "  " << program_name << " -i sample.jpg -pipeline blur,sobel,sharpen" << std::endl;
    std::cout << "  " << program_name << " -i sample.jpg -blur-radius 5 -sobel-threshold 100" << std::endl;
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    std::string image_path;
    std::string pipeline_str;
    int blur_radius = 3;
    float blur_sigma = 1.5f;
    float sobel_threshold = 50.0f;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "-pipeline" && i + 1 < argc) {
            pipeline_str = argv[++i];
        } else if (arg == "-blur-radius" && i + 1 < argc) {
            blur_radius = std::stoi(argv[++i]);
        } else if (arg == "-blur-sigma" && i + 1 < argc) {
            blur_sigma = std::stof(argv[++i]);
        } else if (arg == "-sobel-threshold" && i + 1 < argc) {
            sobel_threshold = std::stof(argv[++i]);
        } else if (arg == "-help" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (image_path.empty()) {
        std::cerr << "Error: Input image path is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        InteractiveImageProcessor processor;
        
        if (!processor.load_image(image_path)) {
            return 1;
        }
        
        // Set default parameters
        processor.set_blur_params(blur_radius, blur_sigma);
        processor.set_sobel_threshold(sobel_threshold);
        
        // Print GPU information
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        
        if (!pipeline_str.empty()) {
            // Batch mode
            std::vector<std::string> filters = split_string(pipeline_str, ',');
            processor.run_batch_mode(filters);
        } else {
            // Interactive mode
            processor.run_interactive_mode();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    cv::destroyAllWindows();
    return 0;
}