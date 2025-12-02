#!/usr/bin/env python3
"""
Build TensorRT engine from ONNX model
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not found. Please install TensorRT.")
    sys.exit(1)


class EngineBuilder:
    """
    TensorRT Engine Builder for YOLOv8
    """
    
    def __init__(self, onnx_path, engine_path, precision='fp16', max_batch_size=1):
        """
        Initialize engine builder
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Output engine path
            precision: Precision mode (fp32/fp16/int8)
            max_batch_size: Maximum batch size
        """
        self.onnx_path = Path(onnx_path)
        self.engine_path = Path(engine_path)
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Verify ONNX exists
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
    
    def build_engine(self, calibrator=None):
        """
        Build TensorRT engine
        
        Args:
            calibrator: INT8 calibration object
        
        Returns:
            Serialized engine bytes
        """
        print(f"Building TensorRT engine from {self.onnx_path}")
        print(f"  Precision: {self.precision.upper()}")
        print(f"  Max batch size: {self.max_batch_size}")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        print("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        print(f"  Network inputs: {network.num_inputs}")
        print(f"  Network outputs: {network.num_outputs}")
        
        # Configure builder
        config = builder.create_builder_config()
        
        # Set max workspace size (8GB)
        config.max_workspace_size = 8 * (1 << 30)
        
        # Set precision
        if self.precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  FP16 mode enabled")
            else:
                print("  Warning: FP16 not supported on this platform")
        elif self.precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator:
                    config.int8_calibrator = calibrator
                    print("  INT8 mode enabled with calibration")
                else:
                    print("  Warning: INT8 mode without calibration")
            else:
                print("  Warning: INT8 not supported on this platform")
        
        # Enable tensor cores if available
        if hasattr(config, 'set_flag'):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            if builder.platform_has_tf32:
                config.set_flag(trt.BuilderFlag.TF32)
                print("  TF32 enabled")
        
        # Build engine
        print("Building engine... (this may take a few minutes)")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build engine")
        
        print(f"Engine built successfully!")
        
        # Print engine info
        self.print_engine_info(engine)
        
        # Serialize engine
        return engine.serialize()
    
    def save_engine(self, engine_bytes):
        """
        Save engine to file
        """
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.engine_path, 'wb') as f:
            f.write(engine_bytes)
        
        file_size = self.engine_path.stat().st_size / 1024 / 1024
        print(f"\nEngine saved to: {self.engine_path}")
        print(f"  File size: {file_size:.2f} MB")
    
    def print_engine_info(self, engine):
        """
        Print engine information
        """
        print("\nEngine Information:")
        print(f"  Bindings: {engine.num_bindings}")
        print(f"  Layers: {engine.num_layers}")
        print(f"  Max batch size: {engine.max_batch_size}")
        
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            
            print(f"  {'Input' if is_input else 'Output'} {i}: {name}")
            print(f"    Shape: {shape}")
            print(f"    Type: {dtype}")


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibration for TensorRT
    """
    
    def __init__(self, calibration_images, batch_size=1, input_shape=(3, 640, 640)):
        """
        Initialize INT8 calibrator
        
        Args:
            calibration_images: Path to calibration images directory
            batch_size: Calibration batch size
            input_shape: Input shape (C, H, W)
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_file = 'calibration.cache'
        
        # Load calibration images
        self.images = self.load_calibration_images(calibration_images)
        self.current_index = 0
        
        # Allocate device memory
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.device_input = cuda.mem_alloc(
            batch_size * np.prod(input_shape) * np.dtype(np.float32).itemsize
        )
    
    def load_calibration_images(self, images_dir):
        """
        Load and preprocess calibration images
        """
        import cv2
        from pathlib import Path
        
        images_dir = Path(images_dir)
        image_paths = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        print(f"Loading {len(image_paths)} calibration images...")
        
        images = []
        for path in image_paths[:100]:  # Use first 100 images
            img = cv2.imread(str(path))
            if img is not None:
                # Preprocess image
                img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)  # HWC to CHW
                img = img.astype(np.float32) / 255.0
                images.append(img)
        
        return np.array(images)
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        """
        Get next batch for calibration
        """
        import pycuda.driver as cuda
        
        if self.current_index >= len(self.images):
            return None
        
        batch = self.images[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Copy to device
        cuda.memcpy_htod(self.device_input, batch.ravel())
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        """
        Read calibration cache if exists
        """
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """
        Write calibration cache
        """
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='yolov8s.engine',
                        help='Output engine path')
    parser.add_argument('--precision', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision mode')
    parser.add_argument('--max-batch-size', type=int, default=1,
                        help='Maximum batch size')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 precision')
    parser.add_argument('--calibration-images', type=str,
                        help='Path to calibration images for INT8')
    
    args = parser.parse_args()
    
    # Determine precision
    if args.int8:
        precision = 'int8'
    elif args.fp16:
        precision = 'fp16'
    else:
        precision = args.precision
    
    # Create builder
    builder = EngineBuilder(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=precision,
        max_batch_size=args.max_batch_size
    )
    
    # Setup INT8 calibrator if needed
    calibrator = None
    if precision == 'int8' and args.calibration_images:
        calibrator = INT8Calibrator(
            calibration_images=args.calibration_images,
            batch_size=args.max_batch_size
        )
    
    # Build and save engine
    try:
        engine_bytes = builder.build_engine(calibrator)
        builder.save_engine(engine_bytes)
        print("\nEngine building complete!")
    except Exception as e:
        print(f"Error building engine: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()