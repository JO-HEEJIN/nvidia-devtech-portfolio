"""
TensorRT Conversion Pipeline for Healthcare VLM Deployment

This module converts ONNX models to highly optimized TensorRT engines for maximum inference performance.
Specialized for medical imaging applications with domain-specific optimizations.

Key Features:
- FP16 and INT8 quantization for medical models
- Dynamic shape profiles for various medical image sizes
- Medical imaging calibration datasets
- GPU memory optimization for healthcare workloads
- Production-ready engine building
"""

import tensorrt as trt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorRTConverter:
    """
    Convert ONNX models to TensorRT engines with healthcare-specific optimizations.
    
    Supports multiple precision modes:
    - FP32: Baseline performance
    - FP16: 2x speedup with minimal accuracy loss
    - INT8: Maximum performance with calibration
    """
    
    def __init__(self, 
                 output_dir: str = "./tensorrt_engines",
                 max_workspace_size: int = 4 * (1 << 30),  # 4GB
                 precision: str = "fp16"):
        """
        Initialize TensorRT converter.
        
        Args:
            output_dir: Directory to save TensorRT engines
            max_workspace_size: Maximum GPU memory for optimization (bytes)
            precision: Precision mode ('fp32', 'fp16', 'int8')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workspace_size = max_workspace_size
        self.precision = precision
        
        # Initialize TensorRT logger
        self.logger = trt.Logger(trt.Logger.INFO)
        
        logger.info(f"TensorRT converter initialized - Precision: {precision}, Workspace: {max_workspace_size//1024//1024}MB")
    
    def convert_onnx_to_tensorrt(self,
                                onnx_path: str,
                                engine_name: str,
                                input_shapes: Dict[str, Tuple[Tuple[int], Tuple[int], Tuple[int]]],
                                calibration_cache: Optional[str] = None) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            engine_name: Name for output engine
            input_shapes: Dict mapping input names to (min_shape, opt_shape, max_shape)
            calibration_cache: Path to INT8 calibration cache
            
        Returns:
            Path to generated TensorRT engine
        """
        logger.info(f"Converting {onnx_path} to TensorRT engine...")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        # Set precision mode
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 optimization enabled")
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_cache:
                config.int8_calibrator = self._create_calibrator(calibration_cache)
            logger.info("INT8 optimization enabled")
        
        # Configure dynamic shapes for medical imaging
        profile = builder.create_optimization_profile()
        for input_name, (min_shape, opt_shape, max_shape) in input_shapes.items():
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            logger.info(f"Dynamic shape configured for {input_name}: {min_shape} -> {opt_shape} -> {max_shape}")
        
        config.add_optimization_profile(profile)
        
        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        build_time = time.time() - start_time
        logger.info(f"Engine built successfully in {build_time:.2f} seconds")
        
        # Save engine
        engine_path = self.output_dir / f"{engine_name}.trt"
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        # Save metadata
        self._save_engine_metadata(engine_path, onnx_path, input_shapes, build_time)
        
        logger.info(f"TensorRT engine saved: {engine_path}")
        return str(engine_path)
    
    def convert_vision_encoder(self,
                              onnx_path: str,
                              engine_name: str = "vision_encoder") -> str:
        """
        Convert vision encoder with medical imaging optimizations.
        
        Medical images vary in size, so we configure dynamic shapes
        for common resolutions used in healthcare:
        - 224x224: Standard for BiomedCLIP
        - 512x512: High-resolution dermoscopy
        - 1024x1024: Digital pathology
        """
        # Define medical imaging input shapes
        input_shapes = {
            "image_input": (
                (1, 3, 224, 224),   # min: standard resolution
                (4, 3, 512, 512),   # opt: common batch + high-res
                (8, 3, 1024, 1024)  # max: large batch + pathology
            )
        }
        
        return self.convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_name=engine_name,
            input_shapes=input_shapes
        )
    
    def convert_text_encoder(self,
                            onnx_path: str,
                            engine_name: str = "text_encoder",
                            max_sequence_length: int = 256) -> str:
        """
        Convert text encoder with medical text optimizations.
        
        Medical text descriptions vary in length:
        - Short: "normal chest x-ray"
        - Medium: detailed radiology reports
        - Long: comprehensive clinical notes
        """
        # Define medical text input shapes
        input_shapes = {
            "text_input": (
                (1, 1),                    # min: single token
                (4, 128),                  # opt: typical medical descriptions
                (16, max_sequence_length)  # max: long reports with large batch
            )
        }
        
        return self.convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            engine_name=engine_name,
            input_shapes=input_shapes
        )
    
    def _create_calibrator(self, calibration_cache_path: str):
        """
        Create INT8 calibrator for medical images.
        Uses medical image dataset for calibration.
        """
        from .quantization import MedicalImageCalibrator
        
        return MedicalImageCalibrator(
            cache_file=calibration_cache_path,
            batch_size=4,
            input_shape=(3, 224, 224)
        )
    
    def _save_engine_metadata(self,
                             engine_path: Path,
                             onnx_path: str,
                             input_shapes: Dict,
                             build_time: float) -> None:
        """Save engine metadata for deployment tracking."""
        metadata = {
            "engine_path": str(engine_path),
            "source_onnx": onnx_path,
            "precision": self.precision,
            "workspace_size_mb": self.max_workspace_size // 1024 // 1024,
            "input_shapes": input_shapes,
            "build_time_seconds": build_time,
            "tensorrt_version": trt.__version__,
            "medical_optimizations": {
                "dynamic_shapes_enabled": True,
                "medical_image_support": True,
                "clinical_text_support": True
            }
        }
        
        metadata_path = engine_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


class TensorRTEngineLoader:
    """
    Load and manage TensorRT engines for inference.
    Handles memory allocation and CUDA context management.
    """
    
    def __init__(self, engine_path: str):
        """
        Initialize engine loader.
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        self._load_engine()
    
    def _load_engine(self) -> None:
        """Load TensorRT engine from file."""
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        
        # Create runtime and load engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Setup I/O bindings
        self._setup_bindings()
        
        # Create CUDA stream for async execution
        self.stream = cuda.Stream()
        
        logger.info("TensorRT engine loaded successfully")
    
    def _setup_bindings(self) -> None:
        """Setup input/output bindings for the engine."""
        self.inputs = []
        self.outputs = []
        self.bindings = [None] * self.engine.num_bindings
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            if self.engine.binding_is_input(i):
                self.inputs.append({'name': name, 'dtype': dtype, 'index': i})
            else:
                self.outputs.append({'name': name, 'dtype': dtype, 'index': i})
    
    def infer(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference with the TensorRT engine.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Set input shapes (for dynamic shapes)
        for input_info in self.inputs:
            input_name = input_info['name']
            if input_name in input_data:
                input_shape = input_data[input_name].shape
                self.context.set_binding_shape(input_info['index'], input_shape)
        
        # Allocate GPU memory
        d_inputs = {}
        d_outputs = {}
        
        # Allocate input buffers
        for input_info in self.inputs:
            input_name = input_info['name']
            if input_name in input_data:
                data = input_data[input_name].astype(input_info['dtype'])
                d_input = cuda.mem_alloc(data.nbytes)
                cuda.memcpy_htod(d_input, data)
                d_inputs[input_name] = d_input
                self.bindings[input_info['index']] = int(d_input)
        
        # Allocate output buffers
        for output_info in self.outputs:
            output_shape = self.context.get_binding_shape(output_info['index'])
            output_size = np.prod(output_shape)
            d_output = cuda.mem_alloc(output_size * np.dtype(output_info['dtype']).itemsize)
            d_outputs[output_info['name']] = d_output
            self.bindings[output_info['index']] = int(d_output)
        
        # Run inference
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        
        # Copy outputs back to host
        results = {}
        for output_info in self.outputs:
            output_name = output_info['name']
            output_shape = self.context.get_binding_shape(output_info['index'])
            h_output = np.empty(output_shape, dtype=output_info['dtype'])
            cuda.memcpy_dtoh(h_output, d_outputs[output_name])
            results[output_name] = h_output
        
        # Synchronize
        self.stream.synchronize()
        
        # Cleanup GPU memory
        for d_input in d_inputs.values():
            d_input.free()
        for d_output in d_outputs.values():
            d_output.free()
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the loaded engine."""
        if self.engine is None:
            return {"status": "No engine loaded"}
        
        return {
            "engine_path": self.engine_path,
            "max_batch_size": self.engine.max_batch_size,
            "num_layers": self.engine.num_layers,
            "workspace_size": self.engine.device_memory_size,
            "has_implicit_batch": self.engine.has_implicit_batch_dimension,
            "inputs": [inp['name'] for inp in self.inputs],
            "outputs": [out['name'] for out in self.outputs]
        }


def convert_biomedclip_to_tensorrt(onnx_dir: str,
                                  output_dir: str = "./tensorrt_engines",
                                  precision: str = "fp16",
                                  max_workspace_size: int = 4 * (1 << 30)) -> Dict[str, str]:
    """
    Convert complete BiomedCLIP ONNX models to TensorRT.
    
    Args:
        onnx_dir: Directory containing ONNX models
        output_dir: Output directory for TensorRT engines
        precision: Precision mode ('fp32', 'fp16', 'int8')
        max_workspace_size: Maximum GPU memory for optimization
        
    Returns:
        Dictionary with paths to converted engines
    """
    converter = TensorRTConverter(
        output_dir=output_dir,
        max_workspace_size=max_workspace_size,
        precision=precision
    )
    
    converted_engines = {}
    onnx_path = Path(onnx_dir)
    
    # Convert vision encoder
    vision_onnx = onnx_path / "vision_encoder.onnx"
    if vision_onnx.exists():
        engine_path = converter.convert_vision_encoder(
            str(vision_onnx),
            f"vision_encoder_{precision}"
        )
        converted_engines['vision_encoder'] = engine_path
    
    # Convert text encoder
    text_onnx = onnx_path / "text_encoder.onnx"
    if text_onnx.exists():
        engine_path = converter.convert_text_encoder(
            str(text_onnx),
            f"text_encoder_{precision}"
        )
        converted_engines['text_encoder'] = engine_path
    
    logger.info(f"TensorRT conversion completed: {converted_engines}")
    return converted_engines


if __name__ == "__main__":
    # Test TensorRT conversion
    try:
        logger.info("Testing TensorRT conversion...")
        
        # Test with dummy ONNX files (would normally be created by export script)
        test_onnx_dir = "./test_onnx_models"
        test_output_dir = "./test_tensorrt_engines"
        
        # This would normally work with real ONNX files
        # engines = convert_biomedclip_to_tensorrt(
        #     onnx_dir=test_onnx_dir,
        #     output_dir=test_output_dir,
        #     precision="fp16"
        # )
        
        logger.info("TensorRT conversion test setup completed")
        
    except Exception as e:
        logger.error(f"TensorRT test failed: {e}")
        logger.info("This is expected without proper TensorRT installation")