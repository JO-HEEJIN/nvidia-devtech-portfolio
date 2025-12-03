#!/usr/bin/env python3
"""
Generate synthetic test images for CUDA image processing benchmarks.
Creates images with various patterns, sizes, and characteristics for testing.
"""

import numpy as np
import cv2
import os
from typing import Tuple

def generate_gradient_image(width: int, height: int, direction: str = 'horizontal') -> np.ndarray:
    """Generate a gradient image."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    if direction == 'horizontal':
        for x in range(width):
            intensity = int(255 * x / width)
            image[:, x] = intensity
    elif direction == 'vertical':
        for y in range(height):
            intensity = int(255 * y / height)
            image[y, :] = intensity
    elif direction == 'radial':
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                intensity = int(255 * (1 - distance / max_distance))
                image[y, x] = max(0, intensity)
    
    return image

def generate_checkerboard(width: int, height: int, square_size: int = 32) -> np.ndarray:
    """Generate a checkerboard pattern."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                image[y, x] = 255
    
    return image

def generate_noise_image(width: int, height: int, noise_type: str = 'gaussian') -> np.ndarray:
    """Generate various types of noise images."""
    if noise_type == 'gaussian':
        # Gaussian noise with mean=128, std=50
        image = np.random.normal(128, 50, (height, width))
        image = np.clip(image, 0, 255).astype(np.uint8)
    elif noise_type == 'uniform':
        # Uniform random noise
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        image = np.full((height, width), 128, dtype=np.uint8)
        # Add salt (white pixels)
        salt_coords = np.random.randint(0, [height, width], (height * width // 20, 2))
        image[salt_coords[:, 0], salt_coords[:, 1]] = 255
        # Add pepper (black pixels)
        pepper_coords = np.random.randint(0, [height, width], (height * width // 20, 2))
        image[pepper_coords[:, 0], pepper_coords[:, 1]] = 0
    
    return image

def generate_circles(width: int, height: int, num_circles: int = 10) -> np.ndarray:
    """Generate image with random circles."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    for _ in range(num_circles):
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(20, min(width, height) // 4)
        intensity = np.random.randint(100, 256)
        
        cv2.circle(image, (center_x, center_y), radius, intensity, -1)
    
    return image

def generate_lines(width: int, height: int, num_lines: int = 15) -> np.ndarray:
    """Generate image with random lines."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    for _ in range(num_lines):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        intensity = np.random.randint(100, 256)
        thickness = np.random.randint(2, 8)
        
        cv2.line(image, (x1, y1), (x2, y2), intensity, thickness)
    
    return image

def generate_text_image(width: int, height: int, text: str = "CUDA TEST") -> np.ndarray:
    """Generate image with text."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 200
    thickness = max(1, int(font_scale * 2))
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Center the text
    x = (width - text_width) // 2
    y = (height + text_height) // 2
    
    cv2.putText(image, text, (x, y), font, font_scale, 255, thickness)
    
    return image

def generate_mandelbrot(width: int, height: int, max_iter: int = 100) -> np.ndarray:
    """Generate Mandelbrot fractal image."""
    # Define complex plane bounds
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25
    
    image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to complex number
            real = x_min + (x / width) * (x_max - x_min)
            imag = y_min + (y / height) * (y_max - y_min)
            c = complex(real, imag)
            
            # Mandelbrot iteration
            z = 0
            for i in range(max_iter):
                if abs(z) > 2:
                    break
                z = z * z + c
            
            # Map iteration count to grayscale value
            intensity = int(255 * i / max_iter)
            image[y, x] = intensity
    
    return image

def add_gaussian_blur(image: np.ndarray, kernel_size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """Add Gaussian blur to image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def add_gaussian_noise(image: np.ndarray, std: float = 25) -> np.ndarray:
    """Add Gaussian noise to existing image."""
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def create_test_images():
    """Create comprehensive test image suite."""
    output_dir = "."
    
    # Standard sizes for testing
    sizes = [
        (640, 480),     # VGA
        (1280, 720),    # HD
        (1920, 1080),   # FHD
        (512, 512),     # Square
        (256, 256),     # Small square
    ]
    
    image_generators = {
        'gradient_horizontal': lambda w, h: generate_gradient_image(w, h, 'horizontal'),
        'gradient_vertical': lambda w, h: generate_gradient_image(w, h, 'vertical'),
        'gradient_radial': lambda w, h: generate_gradient_image(w, h, 'radial'),
        'checkerboard': lambda w, h: generate_checkerboard(w, h, 32),
        'checkerboard_small': lambda w, h: generate_checkerboard(w, h, 8),
        'noise_gaussian': lambda w, h: generate_noise_image(w, h, 'gaussian'),
        'noise_uniform': lambda w, h: generate_noise_image(w, h, 'uniform'),
        'noise_salt_pepper': lambda w, h: generate_noise_image(w, h, 'salt_pepper'),
        'circles': lambda w, h: generate_circles(w, h, 15),
        'lines': lambda w, h: generate_lines(w, h, 20),
        'text': lambda w, h: generate_text_image(w, h),
        'mandelbrot': lambda w, h: generate_mandelbrot(w, h),
    }
    
    print("Generating synthetic test images...")
    
    # Generate images for each size and pattern
    for size_name, (width, height) in zip(['vga', 'hd', 'fhd', 'square', 'small'], sizes):
        print(f"Generating {size_name} ({width}x{height}) images...")
        
        for pattern_name, generator in image_generators.items():
            image = generator(width, height)
            filename = f"test_{pattern_name}_{size_name}_{width}x{height}.png"
            cv2.imwrite(os.path.join(output_dir, filename), image)
    
    # Generate some special purpose images
    print("Generating special purpose images...")
    
    # High contrast edge detection test
    edge_test = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(edge_test, (128, 128), (384, 384), 255, -1)
    cv2.circle(edge_test, (256, 256), 100, 0, -1)
    cv2.imwrite(os.path.join(output_dir, "test_edges_512x512.png"), edge_test)
    
    # Blur test image with sharp features
    blur_test = generate_checkerboard(1024, 768, 16)
    blur_test = add_gaussian_noise(blur_test, 15)
    cv2.imwrite(os.path.join(output_dir, "test_blur_1024x768.png"), blur_test)
    
    # Histogram test with known distribution
    hist_test = np.zeros((512, 512), dtype=np.uint8)
    # Create four regions with different intensities
    hist_test[:256, :256] = 64   # Dark
    hist_test[:256, 256:] = 128  # Medium-dark
    hist_test[256:, :256] = 192  # Medium-bright
    hist_test[256:, 256:] = 255  # Bright
    cv2.imwrite(os.path.join(output_dir, "test_histogram_512x512.png"), hist_test)
    
    # Resize test with fine details
    resize_test = generate_mandelbrot(800, 600, 150)
    cv2.imwrite(os.path.join(output_dir, "test_resize_800x600.png"), resize_test)
    
    # Performance test image (large)
    if True:  # Set to False to skip large image generation
        print("Generating large performance test image (4K)...")
        perf_test = generate_noise_image(3840, 2160, 'gaussian')
        cv2.imwrite(os.path.join(output_dir, "test_performance_4k_3840x2160.png"), perf_test)
    
    print(f"Test image generation complete! Check the current directory for generated images.")

def create_readme():
    """Create README file explaining the test images."""
    readme_content = """# Test Images for CUDA Image Processing

This directory contains synthetic test images generated for benchmarking and testing
CUDA image processing kernels.

## Image Categories

### Size Variants
- **VGA (640x480)**: Standard definition test images
- **HD (1280x720)**: High definition test images  
- **FHD (1920x1080)**: Full HD test images
- **Square (512x512)**: Square format for convolution tests
- **Small (256x256)**: Small images for quick testing

### Pattern Types

#### Gradient Images
- `gradient_horizontal_*`: Horizontal intensity gradient (0->255)
- `gradient_vertical_*`: Vertical intensity gradient (0->255)
- `gradient_radial_*`: Radial gradient from center

#### Geometric Patterns
- `checkerboard_*`: Regular checkerboard pattern (32px squares)
- `checkerboard_small_*`: Fine checkerboard pattern (8px squares)
- `circles_*`: Random circles with varying intensities
- `lines_*`: Random lines for edge detection testing

#### Noise Patterns
- `noise_gaussian_*`: Gaussian distributed noise
- `noise_uniform_*`: Uniform random noise
- `noise_salt_pepper_*`: Salt and pepper noise

#### Special Images
- `text_*`: Text rendering test
- `mandelbrot_*`: Mandelbrot fractal (fine detail test)

### Specialized Test Images

#### `test_edges_512x512.png`
High contrast geometric shapes for edge detection testing.

#### `test_blur_1024x768.png`
Checkerboard with noise, ideal for blur algorithm testing.

#### `test_histogram_512x512.png`
Four-quadrant image with known intensity distribution for histogram validation.

#### `test_resize_800x600.png`
Mandelbrot fractal with fine details for resize algorithm testing.

#### `test_performance_4k_3840x2160.png`
Large 4K image for performance benchmarking (if enabled).

## Usage

Use these images with the CUDA image processing tools:

```bash
# Test individual operations
./cuda_image_processor -i sample_images/test_gradient_horizontal_hd_1280x720.png -o output.png -op blur

# Benchmark with specific image
./benchmark -i sample_images/test_noise_gaussian_fhd_1920x1080.png

# Interactive demo
./demo -i sample_images/test_mandelbrot_square_512x512.png
```

## Regenerating Images

To regenerate all test images:

```bash
cd sample_images
python3 generate_test_images.py
```

Requirements: OpenCV Python (`pip install opencv-python numpy`)
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)

if __name__ == "__main__":
    # Check if OpenCV is available
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("Error: OpenCV and NumPy are required to generate test images.")
        print("Install with: pip install opencv-python numpy")
        exit(1)
    
    create_test_images()
    create_readme()