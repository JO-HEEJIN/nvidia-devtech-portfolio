#!/usr/bin/env python3
"""
ONNX to TensorRT Engine Builder

This module converts ONNX models to optimized TensorRT engines with support
for multiple precision modes (FP32, FP16, INT8). TensorRT performs graph
optimizations including layer fusion, kernel auto-tuning, and precision
calibration to maximize inference performance on NVIDIA GPUs.

Key optimizations performed by TensorRT:
- Layer & tensor fusion: Combines ops into single kernels
- Precision calibration: Mixed precision for optimal speed/accuracy
- Kernel auto-tuning: Selects fastest kernels for target GPU
- Dynamic tensor memory: Efficient memory allocation
- Multi-stream execution: Overlapping compute and memory ops
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import json

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context automatically
from coloredlogs import install as setup_colored_logs

# Import calibration module (to be created)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EngineBuilder:
    """
    TensorRT engine builder with comprehensive optimization options.
    
    This class encapsulates the TensorRT build process, handling:
    - Network definition from ONNX
    - Builder configuration for different precisions
    - Optimization profiles for dynamic shapes
    - Engine serialization
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize TensorRT builder components.
        
        Args:
            verbose: Enable detailed TensorRT logging
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create TensorRT logger
        trt_logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        self.trt_logger = trt.Logger(trt_logger_level)
        
        # Initialize builder and associated objects
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.network = None
        self.parser = None
        
        self.logger.info(f"TensorRT version: {trt.__version__}")
        self.logger.info(f"CUDA compute capability: {cuda.Device(0).compute_capability()}")
        
    def load_onnx(self, onnx_path: str) -> bool:
        """
        Load and parse ONNX model into TensorRT network.
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            True if loading successful, False otherwise
        """
        self.logger.info(f"Loading ONNX model: {onnx_path}")
        
        # Create network with explicit batch dimension
        # EXPLICIT_BATCH flag is required for dynamic batch support
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        
        # Parse ONNX file
        with open(onnx_path, 'rb') as f:
            onnx_data = f.read()
            
        if not self.parser.parse(onnx_data):
            # Print parsing errors
            for i in range(self.parser.num_errors):
                error = self.parser.get_error(i)
                self.logger.error(f"ONNX parsing error: {error}")
            return False
            
        self.logger.info(f"ONNX model loaded successfully")
        self.logger.info(f"Network layers: {self.network.num_layers}")
        self.logger.info(f"Network inputs: {self.network.num_inputs}")
        self.logger.info(f"Network outputs: {self.network.num_outputs}")
        
        # Log input/output dimensions
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            self.logger.info(f"Input {i}: {input_tensor.name}, shape: {input_tensor.shape}")
            
        for i in range(self.network.num_outputs):
            output_tensor = self.network.get_output(i)
            self.logger.info(f"Output {i}: {output_tensor.name}, shape: {output_tensor.shape}")
            
        return True
    
    def configure_builder(
        self,
        precision: str = 'fp32',
        max_workspace_size: int = 1024,
        max_batch_size: int = 16,
        calibrator=None
    ) -> None:
        """
        Configure TensorRT builder with optimization settings.
        
        The workspace size determines how much GPU memory TensorRT can use
        for layer implementation selection and kernel tuning. Larger workspace
        may enable more optimizations but increases memory usage.
        
        Args:
            precision: Target precision mode (fp32/fp16/int8)
            max_workspace_size: Maximum GPU memory for optimization (MB)
            max_batch_size: Maximum batch size for optimization
            calibrator: INT8 calibrator object (required for int8 mode)
        """
        self.logger.info(f"Configuring builder for {precision.upper()} precision")
        
        # Set workspace size (convert MB to bytes)
        self.config.max_workspace_size = max_workspace_size * 1024 * 1024
        
        # Configure precision mode
        if precision.lower() == 'fp16':
            if not self.builder.platform_has_fast_fp16:
                self.logger.warning("Platform doesn't have fast FP16 support!")
            self.config.set_flag(trt.BuilderFlag.FP16)
            self.logger.info("FP16 precision enabled")
            
        elif precision.lower() == 'int8':
            if not self.builder.platform_has_fast_int8:
                self.logger.warning("Platform doesn't have fast INT8 support!")
            self.config.set_flag(trt.BuilderFlag.INT8)
            
            if calibrator is None:
                self.logger.error("INT8 mode requires a calibrator")
                raise ValueError("Calibrator required for INT8 precision")
                
            self.config.int8_calibrator = calibrator
            self.logger.info("INT8 precision enabled with calibrator")
            
        else:  # fp32 (default)
            self.logger.info("Using FP32 precision (default)")
        
        # Additional optimization flags
        # PREFER_PRECISION_CONSTRAINTS: Prefer precision constraints over performance
        # DISABLE_TIMING_CACHE: Disable timing cache for reproducible builds
        # TF32: Use TensorFloat-32 for better performance on Ampere GPUs
        if hasattr(trt.BuilderFlag, 'TF32'):
            self.config.set_flag(trt.BuilderFlag.TF32)
            self.logger.info("TF32 enabled for Ampere GPUs")
            
        # Set optimization profiles for dynamic shapes
        self._setup_optimization_profiles(max_batch_size)
        
        self.logger.info(f"Max workspace size: {max_workspace_size} MB")
        
    def _setup_optimization_profiles(self, max_batch_size: int) -> None:
        """
        Configure optimization profiles for dynamic input shapes.
        
        Optimization profiles define min/opt/max dimensions for dynamic inputs,
        allowing TensorRT to optimize for different batch sizes at runtime.
        
        Args:
            max_batch_size: Maximum batch size to optimize for
        """
        if self.network.num_inputs == 0:
            return
            
        profile = self.builder.create_optimization_profile()
        
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            input_shape = input_tensor.shape
            
            # Check if batch dimension is dynamic (-1)
            if input_shape[0] == -1:
                # Create min, opt, max shapes for dynamic batch
                shape_min = (1, *input_shape[1:])
                shape_opt = (max_batch_size // 2, *input_shape[1:])  # Optimize for mid-range
                shape_max = (max_batch_size, *input_shape[1:])
                
                profile.set_shape(
                    input_tensor.name,
                    shape_min,
                    shape_opt,
                    shape_max
                )
                
                self.logger.info(f"Dynamic shape profile for {input_tensor.name}:")
                self.logger.info(f"  Min: {shape_min}")
                self.logger.info(f"  Opt: {shape_opt}")
                self.logger.info(f"  Max: {shape_max}")
                
        self.config.add_optimization_profile(profile)
        
    def build_engine(self) -> Optional[trt.ICudaEngine]:
        """
        Build optimized TensorRT engine from network definition.
        
        This process performs all optimizations including:
        - Graph optimization and layer fusion
        - Kernel selection and auto-tuning
        - Memory optimization
        - Precision calibration
        
        Returns:
            Built TensorRT engine or None if build fails
        """
        if self.network is None:
            self.logger.error("Network not loaded. Call load_onnx first.")
            return None
            
        self.logger.info("Building TensorRT engine... This may take a while.")
        self.logger.info("TensorRT is performing the following optimizations:")
        self.logger.info("  - Layer & tensor fusion")
        self.logger.info("  - Kernel auto-tuning")
        self.logger.info("  - Memory optimization")
        self.logger.info("  - Precision calibration")
        
        # Build engine
        engine = self.builder.build_engine(self.network, self.config)
        
        if engine is None:
            self.logger.error("Engine build failed!")
            return None
            
        self.logger.info("Engine built successfully!")
        self.logger.info(f"Engine size: {engine.device_memory_size / 1024 / 1024:.2f} MB")
        self.logger.info(f"Number of bindings: {engine.num_bindings}")
        
        # Log binding information
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            
            self.logger.info(
                f"Binding {i}: {name}, "
                f"{'Input' if is_input else 'Output'}, "
                f"dtype: {dtype}, shape: {shape}"
            )
            
        return engine
        
    def save_engine(self, engine: trt.ICudaEngine, output_path: str) -> None:
        """
        Serialize and save TensorRT engine to disk.
        
        Serialized engines can be loaded directly for inference without
        rebuilding, significantly reducing deployment time.
        
        Args:
            engine: Built TensorRT engine
            output_path: Path to save serialized engine
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
            
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
        self.logger.info(f"Engine saved: {output_path} ({file_size:.2f} MB)")
        
        # Save engine metadata
        metadata = {
            'tensorrt_version': trt.__version__,
            'cuda_compute_capability': str(cuda.Device(0).compute_capability()),
            'engine_size_mb': file_size,
            'num_bindings': engine.num_bindings,
            'bindings': []
        }
        
        for i in range(engine.num_bindings):
            binding_info = {
                'name': engine.get_binding_name(i),
                'dtype': str(engine.get_binding_dtype(i)),
                'shape': list(engine.get_binding_shape(i)),
                'is_input': engine.binding_is_input(i)
            }
            metadata['bindings'].append(binding_info)
            
        metadata_path = output_path.replace('.trt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Metadata saved: {metadata_path}")


def create_int8_calibrator(
    calibration_data_dir: str,
    cache_file: str = 'calibration.cache',
    batch_size: int = 8,
    max_batches: int = 10
):
    """
    Create INT8 calibrator for quantization.
    
    The calibrator collects activation statistics during calibration
    to determine optimal quantization thresholds for each layer.
    
    Args:
        calibration_data_dir: Directory containing calibration images
        cache_file: Path to cache calibration results
        batch_size: Batch size for calibration
        max_batches: Maximum number of batches to process
        
    Returns:
        Calibrator object for INT8 quantization
    """
    # Import calibration module
    try:
        from calibration import INT8EntropyCalibrator
        return INT8EntropyCalibrator(
            calibration_data_dir,
            cache_file,
            batch_size,
            max_batches
        )
    except ImportError:
        logging.warning("Calibration module not found. Using dummy calibrator.")
        return None


def main():
    """Main entry point for TensorRT engine building."""
    
    parser = argparse.ArgumentParser(
        description='Convert ONNX models to optimized TensorRT engines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--onnx',
        type=str,
        required=True,
        help='Input ONNX model path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output TensorRT engine path'
    )
    
    parser.add_argument(
        '--precision',
        type=str,
        default='fp32',
        choices=['fp32', 'fp16', 'int8'],
        help='Precision mode for optimization'
    )
    
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=16,
        help='Maximum batch size for optimization'
    )
    
    parser.add_argument(
        '--workspace-size',
        type=int,
        default=1024,
        help='Maximum workspace size in MB'
    )
    
    parser.add_argument(
        '--calibration-data',
        type=str,
        help='Directory containing calibration images (required for INT8)'
    )
    
    parser.add_argument(
        '--calibration-cache',
        type=str,
        default='calibration.cache',
        help='Cache file for INT8 calibration'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_colored_logs(
        level=level,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("TensorRT Engine Builder")
    logger.info("="*60)
    
    try:
        # Create builder
        builder = EngineBuilder(verbose=args.verbose)
        
        # Load ONNX model
        if not builder.load_onnx(args.onnx):
            logger.error("Failed to load ONNX model")
            sys.exit(1)
            
        # Create calibrator for INT8 if needed
        calibrator = None
        if args.precision == 'int8':
            if not args.calibration_data:
                logger.error("Calibration data directory required for INT8 precision")
                sys.exit(1)
                
            calibrator = create_int8_calibrator(
                args.calibration_data,
                args.calibration_cache
            )
            
            if calibrator is None:
                logger.error("Failed to create INT8 calibrator")
                sys.exit(1)
                
        # Configure builder
        builder.configure_builder(
            precision=args.precision,
            max_workspace_size=args.workspace_size,
            max_batch_size=args.max_batch_size,
            calibrator=calibrator
        )
        
        # Build engine
        engine = builder.build_engine()
        if engine is None:
            logger.error("Engine build failed")
            sys.exit(1)
            
        # Save engine
        builder.save_engine(engine, args.output)
        
        logger.info("="*60)
        logger.info("Engine build completed successfully!")
        logger.info(f"Engine saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Engine build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()