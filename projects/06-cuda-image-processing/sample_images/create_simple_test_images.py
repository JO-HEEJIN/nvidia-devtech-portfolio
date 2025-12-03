#!/usr/bin/env python3
"""
Simple test image generator without external dependencies.
Creates basic test images using only standard library.
"""

def create_pgm_image(filename, width, height, data):
    """Create a PGM (Portable Graymap) image file."""
    with open(filename, 'wb') as f:
        # PGM header
        f.write(f"P5\n{width} {height}\n255\n".encode('ascii'))
        # Image data
        f.write(bytes(data))

def generate_gradient_horizontal(width, height):
    """Generate horizontal gradient."""
    data = []
    for y in range(height):
        for x in range(width):
            intensity = int(255 * x / width)
            data.append(intensity)
    return data

def generate_gradient_vertical(width, height):
    """Generate vertical gradient."""
    data = []
    for y in range(height):
        for x in range(width):
            intensity = int(255 * y / height)
            data.append(intensity)
    return data

def generate_checkerboard(width, height, square_size=32):
    """Generate checkerboard pattern."""
    data = []
    for y in range(height):
        for x in range(width):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                data.append(255)
            else:
                data.append(0)
    return data

def generate_noise(width, height, seed=12345):
    """Generate pseudo-random noise."""
    # Simple linear congruential generator
    import random
    random.seed(seed)
    
    data = []
    for _ in range(width * height):
        data.append(random.randint(0, 255))
    return data

def generate_circles(width, height):
    """Generate image with circles."""
    data = [0] * (width * height)
    
    # Simple circle drawing
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    for y in range(height):
        for x in range(width):
            distance_sq = (x - center_x)**2 + (y - center_y)**2
            if distance_sq <= radius**2:
                data[y * width + x] = 255
            elif distance_sq <= (radius + 20)**2:
                data[y * width + x] = 128
    
    return data

def generate_test_images():
    """Generate basic test images."""
    print("Creating simple test images...")
    
    # Common sizes
    sizes = [
        ("small", 256, 256),
        ("medium", 512, 512),
        ("hd", 1280, 720),
    ]
    
    generators = [
        ("gradient_h", generate_gradient_horizontal),
        ("gradient_v", generate_gradient_vertical),
        ("checkerboard", lambda w, h: generate_checkerboard(w, h, 32)),
        ("noise", generate_noise),
        ("circles", generate_circles),
    ]
    
    for size_name, width, height in sizes:
        for pattern_name, generator in generators:
            filename = f"test_{pattern_name}_{size_name}_{width}x{height}.pgm"
            print(f"Generating {filename}...")
            data = generator(width, height)
            create_pgm_image(filename, width, height, data)
    
    # Create a special edge detection test image
    print("Generating edge detection test image...")
    width, height = 512, 512
    data = [0] * (width * height)
    
    # Add rectangle
    for y in range(128, 384):
        for x in range(128, 384):
            data[y * width + x] = 255
    
    # Add circle (subtract from rectangle)
    center_x, center_y = 256, 256
    radius = 80
    for y in range(height):
        for x in range(width):
            distance_sq = (x - center_x)**2 + (y - center_y)**2
            if distance_sq <= radius**2:
                data[y * width + x] = 0
    
    create_pgm_image("test_edges_512x512.pgm", width, height, data)
    
    print("Created test images in PGM format.")
    print("Note: PGM files can be converted to other formats using ImageMagick or GIMP.")
    print("Example: convert test_gradient_h_medium_512x512.pgm test_gradient_h_medium_512x512.png")

if __name__ == "__main__":
    generate_test_images()