# Test Images for CUDA Image Processing

This directory contains synthetic test images for benchmarking and testing CUDA image processing kernels.

## Generated Images

### Size Variants
- **Small (256x256)**: Quick testing and development
- **Medium (512x512)**: Standard testing size
- **HD (1280x720)**: High definition performance testing

### Pattern Types

#### Gradient Images
- `test_gradient_h_*`: Horizontal intensity gradient (0→255)
- `test_gradient_v_*`: Vertical intensity gradient (0→255)
- Good for testing memory access patterns and interpolation

#### Geometric Patterns
- `test_checkerboard_*`: Regular checkerboard pattern
- High contrast pattern ideal for edge detection and convolution testing

#### Noise Images  
- `test_noise_*`: Pseudo-random noise patterns
- Tests performance with unpredictable data access patterns

#### Shape Images
- `test_circles_*`: Circular patterns with gradients
- Tests radial patterns and smooth gradients

#### Special Test Images
- `test_edges_512x512.pgm`: High contrast shapes for edge detection validation

## File Format

Images are generated in PGM (Portable Graymap) format, which is:
- Simple uncompressed grayscale format
- Directly readable by most image processing libraries
- Easy to convert to other formats

## Converting Images

To convert PGM files to PNG/JPEG for easier viewing:

```bash
# Using ImageMagick (if available)
convert test_gradient_h_medium_512x512.pgm test_gradient_h_medium_512x512.png

# Using GIMP (GUI)
# Open PGM file in GIMP and export as desired format
```

## Usage Examples

```bash
# Test with small image for quick iteration
./cuda_image_processor -i sample_images/test_gradient_h_small_256x256.pgm -o output.pgm -op blur

# Performance testing with HD image
./benchmark -i sample_images/test_noise_hd_1280x720.pgm

# Edge detection testing
./cuda_image_processor -i sample_images/test_edges_512x512.pgm -o edges.pgm -op sobel

# Interactive demo
./demo -i sample_images/test_checkerboard_medium_512x512.pgm
```

## Regenerating Images

To regenerate test images:

```bash
cd sample_images
python3 create_simple_test_images.py
```

Or for enhanced images with more patterns (requires numpy/opencv):

```bash
python3 generate_test_images.py  # If dependencies available
```

## Image Characteristics

### Memory Access Patterns
- **Gradient images**: Sequential patterns, good for cache testing
- **Checkerboard**: Regular patterns with high contrast
- **Noise**: Random patterns, worst-case for cache performance
- **Circles**: Radial patterns, tests non-linear access

### Processing Characteristics
- **Small images**: Fast iteration, CPU vs GPU overhead visible
- **Medium images**: Balanced testing, typical workloads
- **HD images**: GPU advantage clear, memory bandwidth limited

### Validation
- **Gradient images**: Easy to verify output correctness
- **Checkerboard**: Clear pattern recognition for filter effects
- **Noise**: Statistical properties for histogram testing
- **Edges test**: Known geometric shapes for edge detection validation