#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

// Image structure for unified handling
struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;
    size_t pitch; // For aligned memory
    
    Image() : data(nullptr), width(0), height(0), channels(0), pitch(0) {}
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        pitch = width * channels;
        data = new unsigned char[height * pitch];
    }
    
    ~Image() {
        delete[] data;
    }
    
    // Disable copy constructor and assignment to prevent double deletion
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    
    // Move constructor and assignment
    Image(Image&& other) noexcept 
        : data(other.data), width(other.width), height(other.height), 
          channels(other.channels), pitch(other.pitch) {
        other.data = nullptr;
    }
    
    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            width = other.width;
            height = other.height;
            channels = other.channels;
            pitch = other.pitch;
            other.data = nullptr;
        }
        return *this;
    }
    
    size_t size_bytes() const {
        return height * pitch;
    }
    
    bool is_valid() const {
        return data != nullptr && width > 0 && height > 0 && channels > 0;
    }
};

// Load image from file using OpenCV
std::unique_ptr<Image> load_image(const std::string& filename);

// Save image to file using OpenCV
bool save_image(const std::string& filename, const Image& image);

// Convert OpenCV Mat to Image structure
std::unique_ptr<Image> mat_to_image(const cv::Mat& mat);

// Convert Image structure to OpenCV Mat
cv::Mat image_to_mat(const Image& image);

// Create test images
std::unique_ptr<Image> create_test_checkerboard(int width, int height, int square_size = 32);
std::unique_ptr<Image> create_test_gradient(int width, int height, bool horizontal = true);
std::unique_ptr<Image> create_test_noise(int width, int height, float noise_level = 0.1f);

// Format conversion utilities
std::unique_ptr<Image> convert_to_grayscale_cpu(const Image& rgb_image);
std::unique_ptr<Image> convert_to_rgb(const Image& grayscale_image);

// Image validation and information
bool validate_image_format(const std::string& filename);
void print_image_info(const Image& image, const std::string& name = "Image");

// Pixel access helpers (with bounds checking)
inline unsigned char get_pixel(const Image& image, int x, int y, int channel = 0) {
    if (x < 0 || x >= image.width || y < 0 || y >= image.height || channel >= image.channels) {
        return 0;
    }
    return image.data[y * image.pitch + x * image.channels + channel];
}

inline void set_pixel(Image& image, int x, int y, int channel, unsigned char value) {
    if (x < 0 || x >= image.width || y < 0 || y >= image.height || channel >= image.channels) {
        return;
    }
    image.data[y * image.pitch + x * image.channels + channel] = value;
}

// Memory alignment helpers for CUDA
inline size_t align_size(size_t size, size_t alignment = 256) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Create aligned image for optimal CUDA performance
std::unique_ptr<Image> create_aligned_image(int width, int height, int channels, size_t alignment = 256);

// Copy image data with potential format conversion
void copy_image_data(const Image& src, Image& dst, bool convert_format = false);

#endif // IMAGE_IO_H