#!/usr/bin/env python3
"""
INT8 Calibration for TensorRT Quantization

This module implements calibration for INT8 quantization in TensorRT.
INT8 quantization reduces model size by 75% and can provide 2-4x speedup
with minimal accuracy loss when properly calibrated.

The calibration process:
1. Collects activation statistics from representative data
2. Determines optimal quantization thresholds for each layer
3. Caches results for faster engine rebuilds
4. Uses entropy calibration to minimize information loss

Key concepts:
- Calibration dataset should represent real inference data distribution
- More calibration data generally improves accuracy
- Cache files speed up iterative optimization
- Different calibrators trade off speed vs accuracy
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import struct

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import cv2


class CalibrationDataLoader:
    """
    Data loader for calibration images with preprocessing.
    
    This class handles loading and preprocessing of calibration data,
    ensuring images are in the correct format for network input.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        max_batches: int = 10,
        preprocessing: str = 'imagenet'
    ):
        """
        Initialize calibration data loader.
        
        Args:
            data_dir: Directory containing calibration images
            batch_size: Number of images per batch
            input_shape: Expected input shape (C, H, W)
            max_batches: Maximum number of batches to process
            preprocessing: Preprocessing mode ('imagenet' or 'normalize')
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_batches = max_batches
        self.preprocessing = preprocessing
        
        # Collect image paths
        self.image_paths = self._collect_images()
        self.num_images = len(self.image_paths)
        self.num_batches = min(
            self.num_images // batch_size,
            max_batches
        )
        
        self.logger.info(f"Found {self.num_images} calibration images")
        self.logger.info(f"Will process {self.num_batches} batches")
        
        self.batch_idx = 0
        
    def _collect_images(self) -> List[Path]:
        """
        Collect all image paths from the data directory.
        
        Returns:
            List of image file paths
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for ext in supported_formats:
            image_paths.extend(self.data_dir.glob(f'*{ext}'))
            image_paths.extend(self.data_dir.glob(f'*{ext.upper()}'))
            
        return sorted(image_paths)
        
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image for calibration.
        
        Preprocessing steps:
        1. Load and resize to network input size
        2. Convert to RGB if needed
        3. Transpose to CHW format (channels first)
        4. Apply normalization based on preprocessing mode
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        channels, height, width = self.input_shape
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize with antialiasing for better quality
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image, dtype=np.float32)
        
        # Apply preprocessing based on mode
        if self.preprocessing == 'imagenet':
            # ImageNet preprocessing: subtract mean, divide by std
            # These are the standard ImageNet normalization values
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            image_np = (image_np - mean) / std
        elif self.preprocessing == 'normalize':
            # Simple normalization to [0, 1]
            image_np = image_np / 255.0
        else:
            # No preprocessing (raw pixel values)
            pass
            
        # Transpose from HWC to CHW format
        image_np = image_np.transpose(2, 0, 1)
        
        return image_np
        
    def get_batch(self) -> Optional[np.ndarray]:
        """
        Get the next batch of preprocessed images.
        
        Returns:
            Batch of images or None if no more batches
        """
        if self.batch_idx >= self.num_batches:
            return None
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        
        batch_paths = self.image_paths[start_idx:end_idx]
        batch_data = []
        
        for path in batch_paths:
            try:
                image = self.preprocess_image(path)
                batch_data.append(image)
            except Exception as e:
                self.logger.warning(f"Failed to process {path}: {e}")
                # Use black image as fallback
                batch_data.append(np.zeros(self.input_shape, dtype=np.float32))
                
        self.batch_idx += 1
        
        # Stack into batch array
        return np.stack(batch_data, axis=0)
        
    def reset(self):
        """Reset the data loader to start from the beginning."""
        self.batch_idx = 0


class INT8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator using entropy calibration algorithm.
    
    Entropy calibration (IInt8EntropyCalibrator2) minimizes the
    information loss between FP32 and INT8 representations by
    finding optimal quantization thresholds based on entropy.
    
    This is generally the recommended calibrator for CNNs as it
    provides good accuracy with reasonable calibration time.
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_file: str = 'calibration.cache',
        batch_size: int = 8,
        max_batches: int = 10,
        input_shape: Tuple[int, int, int] = (3, 224, 224)
    ):
        """
        Initialize INT8 entropy calibrator.
        
        Args:
            data_dir: Directory containing calibration images
            cache_file: Path to cache calibration results
            batch_size: Batch size for calibration
            max_batches: Maximum batches to process
            input_shape: Network input shape (C, H, W)
        """
        super().__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_file = cache_file
        self.batch_size = batch_size
        
        # Initialize data loader
        self.data_loader = CalibrationDataLoader(
            data_dir,
            batch_size,
            input_shape,
            max_batches
        )
        
        # Calculate buffer size
        self.input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(batch_size * self.input_size)
        
        self.logger.info(f"Calibrator initialized with batch size {batch_size}")
        self.logger.info(f"Cache file: {cache_file}")
        
    def get_batch_size(self) -> int:
        """
        Return the batch size used for calibration.
        
        TensorRT calls this to determine batch size for calibration.
        
        Returns:
            Batch size
        """
        return self.batch_size
        
    def get_batch(self, names: List[str]) -> List[int]:
        """
        Get next batch of calibration data.
        
        This method is called by TensorRT during calibration to get
        input data for collecting activation statistics.
        
        Args:
            names: List of input names (unused but required by API)
            
        Returns:
            List of device memory pointers or None if no more data
        """
        batch = self.data_loader.get_batch()
        
        if batch is None:
            return None
            
        # Copy batch to device memory
        cuda.memcpy_htod(self.device_input, batch.astype(np.float32).ravel())
        
        # Return device memory pointer
        # TensorRT expects a list even for single input
        return [int(self.device_input)]
        
    def read_calibration_cache(self) -> bytes:
        """
        Read calibration cache from file if it exists.
        
        Cached calibration data speeds up engine rebuilds by avoiding
        the need to recalibrate. The cache contains quantization
        thresholds for each tensor in the network.
        
        Returns:
            Cached calibration data or None if cache doesn't exist
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                cache_data = f.read()
                self.logger.info(f"Loaded calibration cache from {self.cache_file}")
                return cache_data
        else:
            self.logger.info("No calibration cache found, will calibrate from scratch")
            return None
            
    def write_calibration_cache(self, cache: bytes):
        """
        Write calibration results to cache file.
        
        Args:
            cache: Calibration data to cache
        """
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        self.logger.info(f"Calibration cache saved to {self.cache_file}")
        
    def free(self):
        """Free allocated device memory."""
        self.device_input.free()


class INT8MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    """
    INT8 calibrator using min-max algorithm.
    
    MinMax calibration uses the minimum and maximum values observed
    during calibration to determine quantization thresholds. This is
    faster than entropy calibration but may be less accurate.
    
    Best for: Networks where speed is critical and slight accuracy
    loss is acceptable.
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_file: str = 'calibration_minmax.cache',
        batch_size: int = 8,
        max_batches: int = 10,
        input_shape: Tuple[int, int, int] = (3, 224, 224)
    ):
        """Initialize MinMax calibrator with same interface as entropy."""
        super().__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_file = cache_file
        self.batch_size = batch_size
        
        # Initialize data loader
        self.data_loader = CalibrationDataLoader(
            data_dir,
            batch_size,
            input_shape,
            max_batches,
            preprocessing='normalize'  # MinMax often works better with [0,1] range
        )
        
        # Calculate buffer size
        self.input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(batch_size * self.input_size)
        
        self.logger.info("Using MinMax calibration (faster but potentially less accurate)")
        
    def get_batch_size(self) -> int:
        """Return batch size for calibration."""
        return self.batch_size
        
    def get_batch(self, names: List[str]) -> List[int]:
        """Get next calibration batch."""
        batch = self.data_loader.get_batch()
        
        if batch is None:
            return None
            
        cuda.memcpy_htod(self.device_input, batch.astype(np.float32).ravel())
        return [int(self.device_input)]
        
    def read_calibration_cache(self) -> bytes:
        """Read calibration cache."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
        
    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
            
    def free(self):
        """Free device memory."""
        self.device_input.free()


def generate_calibration_data(
    output_dir: str,
    num_images: int = 1000,
    image_size: Tuple[int, int] = (224, 224)
):
    """
    Generate synthetic calibration images for testing.
    
    Creates random images that simulate ImageNet-like data distribution
    for testing the calibration pipeline when real data isn't available.
    
    Args:
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        image_size: Size of generated images (H, W)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_images} calibration images...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Generate random image with ImageNet-like statistics
        # Use different patterns to simulate variety
        if i % 3 == 0:
            # Random noise
            image = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        elif i % 3 == 1:
            # Gradient patterns
            x = np.linspace(0, 255, image_size[0])
            y = np.linspace(0, 255, image_size[1])
            xx, yy = np.meshgrid(x, y)
            image = np.stack([xx, yy, (xx + yy) / 2], axis=2).astype(np.uint8)
        else:
            # Solid colors with noise
            base_color = np.random.randint(0, 256, 3)
            noise = np.random.randn(*image_size, 3) * 30
            image = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            
        # Save image
        img_path = output_path / f'calibration_{i:04d}.jpg'
        Image.fromarray(image).save(img_path)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_images} images")
            
    logger.info(f"Calibration images saved to {output_dir}")


if __name__ == '__main__':
    """Test calibration data generation."""
    import argparse
    from coloredlogs import install as setup_colored_logs
    
    parser = argparse.ArgumentParser(description='Generate calibration data for INT8 quantization')
    parser.add_argument('--output', default='calibration_images', help='Output directory')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Image size')
    
    args = parser.parse_args()
    
    setup_colored_logs(level=logging.INFO)
    
    generate_calibration_data(
        args.output,
        args.num_images,
        tuple(args.size)
    )