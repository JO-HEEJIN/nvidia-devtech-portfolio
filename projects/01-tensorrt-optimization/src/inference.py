#!/usr/bin/env python3
"""
TensorRT Inference Engine Wrapper

This module provides a high-level interface for TensorRT inference with
proper CUDA memory management, batch processing, and performance optimization.

Key features:
- Efficient CUDA memory management with pinned host memory
- Support for dynamic batch sizes
- Asynchronous execution with CUDA streams
- Warmup utilities for stable benchmarking
- Context managers for resource cleanup
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torch


class TensorRTInferenceEngine:
    """
    High-performance inference engine wrapper for TensorRT.
    
    This class manages:
    - TensorRT engine loading and context creation
    - CUDA memory allocation and data transfers
    - Batch inference with dynamic shapes
    - Performance optimization through streams and pinned memory
    """
    
    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 16,
        use_cuda_graph: bool = False,
        verbose: bool = False
    ):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to serialized TensorRT engine
            max_batch_size: Maximum batch size for memory allocation
            use_cuda_graph: Enable CUDA graphs for additional optimization
            verbose: Enable detailed logging
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.use_cuda_graph = use_cuda_graph
        
        # Initialize TensorRT logger
        trt_logger_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        self.trt_logger = trt.Logger(trt_logger_level)
        
        # Load engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # Get binding information
        self.bindings_info = self._get_bindings_info()
        
        # Allocate memory buffers
        self.buffers = self._allocate_buffers()
        
        # Create CUDA stream for async execution
        self.stream = cuda.Stream()
        
        # Performance metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        self.logger.info(f"Engine loaded from {engine_path}")
        self.logger.info(f"Max batch size: {max_batch_size}")
        
    def _load_engine(self) -> trt.ICudaEngine:
        """
        Load serialized TensorRT engine from disk.
        
        Returns:
            Deserialized TensorRT engine
            
        Raises:
            RuntimeError: If engine cannot be loaded
        """
        self.logger.info(f"Loading TensorRT engine from {self.engine_path}")
        
        runtime = trt.Runtime(self.trt_logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
            
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            raise RuntimeError(f"Failed to load engine from {self.engine_path}")
            
        self.logger.info(f"Engine loaded successfully")
        self.logger.info(f"Number of bindings: {engine.num_bindings}")
        
        return engine
        
    def _get_bindings_info(self) -> Dict[str, Dict]:
        """
        Extract binding information from the engine.
        
        Bindings are the input/output tensors of the network.
        This method collects metadata about each binding for
        memory allocation and data handling.
        
        Returns:
            Dictionary mapping binding names to their properties
        """
        bindings_info = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            is_input = self.engine.binding_is_input(i)
            
            # Convert TensorRT dtype to numpy dtype
            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            elif dtype == trt.DataType.INT8:
                np_dtype = np.int8
            elif dtype == trt.DataType.INT32:
                np_dtype = np.int32
            else:
                np_dtype = np.float32
                
            bindings_info[name] = {
                'index': i,
                'shape': tuple(shape),
                'dtype': np_dtype,
                'is_input': is_input,
                'trt_dtype': dtype
            }
            
            self.logger.info(
                f"Binding {name}: {'Input' if is_input else 'Output'}, "
                f"shape: {shape}, dtype: {np_dtype}"
            )
            
        return bindings_info
        
    def _allocate_buffers(self) -> Dict[str, Dict]:
        """
        Allocate CUDA memory buffers for inference.
        
        Uses pinned host memory for faster CPU-GPU transfers and
        device memory for actual computation.
        
        Returns:
            Dictionary of allocated buffers for each binding
        """
        buffers = {}
        
        for name, info in self.bindings_info.items():
            # Calculate buffer size
            # Handle dynamic batch dimension
            shape = list(info['shape'])
            if shape[0] == -1:
                shape[0] = self.max_batch_size
                
            size = int(np.prod(shape))
            dtype = info['dtype']
            nbytes = size * dtype().itemsize
            
            # Allocate pinned host memory for faster transfers
            host_buffer = cuda.pagelocked_empty(size, dtype)
            
            # Allocate device memory
            device_buffer = cuda.mem_alloc(nbytes)
            
            buffers[name] = {
                'host': host_buffer,
                'device': device_buffer,
                'size': size,
                'nbytes': nbytes,
                'shape': tuple(shape),
                'dtype': dtype
            }
            
            self.logger.info(f"Allocated buffer for {name}: {nbytes} bytes")
            
        return buffers
        
    def set_input_shape(self, batch_size: int) -> None:
        """
        Set dynamic input shape for the current inference.
        
        Required when using dynamic shapes to inform TensorRT
        about the actual dimensions for this inference.
        
        Args:
            batch_size: Actual batch size for this inference
        """
        for name, info in self.bindings_info.items():
            if info['is_input']:
                shape = list(info['shape'])
                if shape[0] == -1:
                    shape[0] = batch_size
                    
                self.context.set_binding_shape(info['index'], tuple(shape))
                
    def preprocess_input(
        self,
        input_data: Union[np.ndarray, torch.Tensor, str, List],
        input_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Preprocess input data for inference.
        
        Handles various input formats:
        - Numpy arrays: Used directly
        - PyTorch tensors: Converted to numpy
        - Image paths: Loaded and preprocessed
        - Lists: Batched together
        
        Args:
            input_data: Input data in various formats
            input_name: Name of input binding (auto-detected if None)
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        # Auto-detect input name if not provided
        if input_name is None:
            input_names = [n for n, i in self.bindings_info.items() if i['is_input']]
            if len(input_names) != 1:
                raise ValueError(f"Multiple inputs found: {input_names}. Specify input_name.")
            input_name = input_names[0]
            
        input_info = self.bindings_info[input_name]
        expected_shape = list(input_info['shape'])
        
        # Handle different input types
        if isinstance(input_data, torch.Tensor):
            input_array = input_data.detach().cpu().numpy()
        elif isinstance(input_data, str):
            # Load image from path
            input_array = self._load_and_preprocess_image(input_data, expected_shape)
        elif isinstance(input_data, list):
            # Batch multiple inputs
            processed = [self.preprocess_input(item, input_name) for item in input_data]
            input_array = np.stack(processed, axis=0)
        else:
            input_array = np.asarray(input_data)
            
        # Ensure correct dtype
        if input_array.dtype != input_info['dtype']:
            input_array = input_array.astype(input_info['dtype'])
            
        return input_array
        
    def _load_and_preprocess_image(
        self,
        image_path: str,
        expected_shape: List[int]
    ) -> np.ndarray:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to image file
            expected_shape: Expected input shape
            
        Returns:
            Preprocessed image array
        """
        # Extract dimensions (assuming NCHW format)
        if len(expected_shape) == 4:
            _, channels, height, width = expected_shape
        else:
            channels, height, width = expected_shape[-3:]
            
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        image_np = np.array(image, dtype=np.float32)
        
        # ImageNet normalization
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image_np = (image_np - mean) / std
        
        # Transpose to CHW format
        image_np = image_np.transpose(2, 0, 1)
        
        # Add batch dimension if needed
        if len(expected_shape) == 4:
            image_np = np.expand_dims(image_np, 0)
            
        return image_np
        
    def infer(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        sync: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on input data.
        
        Performs the following steps:
        1. Copy input data to device
        2. Execute inference
        3. Copy output data back to host
        4. Return results
        
        Args:
            input_data: Input data as array or dict of arrays
            sync: Whether to synchronize after inference
            
        Returns:
            Inference output(s)
        """
        start_time = time.perf_counter()
        
        # Handle single input vs multiple inputs
        if isinstance(input_data, np.ndarray):
            # Single input - auto-detect binding
            input_names = [n for n, i in self.bindings_info.items() if i['is_input']]
            if len(input_names) != 1:
                raise ValueError("Single array provided but multiple inputs exist")
            input_data = {input_names[0]: input_data}
            
        # Get batch size from input
        batch_size = next(iter(input_data.values())).shape[0]
        
        # Set dynamic shape if needed
        self.set_input_shape(batch_size)
        
        # Prepare device pointers array for execution
        device_ptrs = [None] * self.engine.num_bindings
        
        # Copy input data to device
        for name, data in input_data.items():
            if name not in self.bindings_info:
                raise ValueError(f"Unknown input: {name}")
                
            info = self.bindings_info[name]
            buffer = self.buffers[name]
            
            # Flatten and copy to pinned memory
            np.copyto(buffer['host'][:data.size], data.ravel())
            
            # Copy to device asynchronously
            cuda.memcpy_htod_async(
                buffer['device'],
                buffer['host'],
                self.stream
            )
            
            device_ptrs[info['index']] = int(buffer['device'])
            
        # Execute inference
        success = self.context.execute_async_v2(
            bindings=device_ptrs,
            stream_handle=self.stream.handle
        )
        
        if not success:
            raise RuntimeError("Inference execution failed")
            
        # Prepare output dictionary
        outputs = {}
        
        # Copy output data back to host
        for name, info in self.bindings_info.items():
            if not info['is_input']:
                buffer = self.buffers[name]
                
                # Set device pointer if not already set
                if device_ptrs[info['index']] is None:
                    device_ptrs[info['index']] = int(buffer['device'])
                    
                # Copy from device to pinned memory
                cuda.memcpy_dtoh_async(
                    buffer['host'],
                    buffer['device'],
                    self.stream
                )
                
                # Will be reshaped after sync
                outputs[name] = buffer
                
        # Synchronize if requested
        if sync:
            self.stream.synchronize()
            
            # Reshape outputs to correct dimensions
            for name in list(outputs.keys()):
                buffer = outputs[name]
                info = self.bindings_info[name]
                
                # Calculate actual output shape
                output_shape = list(info['shape'])
                if output_shape[0] == -1:
                    output_shape[0] = batch_size
                    
                # Get actual data size
                actual_size = int(np.prod(output_shape))
                
                # Copy and reshape data
                output_data = buffer['host'][:actual_size].reshape(output_shape)
                outputs[name] = output_data.copy()
                
        # Update metrics
        inference_time = time.perf_counter() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Return single output if only one exists
        if len(outputs) == 1:
            return next(iter(outputs.values()))
            
        return outputs
        
    def warmup(self, num_iterations: int = 10) -> None:
        """
        Perform warmup iterations for stable benchmarking.
        
        Warmup is crucial for accurate benchmarking as it:
        - Initializes CUDA contexts
        - Loads kernels into cache
        - Stabilizes GPU clocks
        
        Args:
            num_iterations: Number of warmup iterations
        """
        self.logger.info(f"Running {num_iterations} warmup iterations...")
        
        # Create dummy input
        dummy_inputs = {}
        for name, info in self.bindings_info.items():
            if info['is_input']:
                shape = list(info['shape'])
                if shape[0] == -1:
                    shape[0] = 1  # Use batch size 1 for warmup
                    
                dummy_inputs[name] = np.random.randn(*shape).astype(info['dtype'])
                
        # Run warmup iterations
        for i in range(num_iterations):
            _ = self.infer(dummy_inputs)
            
        # Reset metrics after warmup
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        self.logger.info("Warmup complete")
        
    def benchmark(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_data: Input data for benchmarking
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with performance metrics
        """
        # Warmup
        if warmup_iterations > 0:
            self.warmup(warmup_iterations)
            
        # Benchmark
        latencies = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.infer(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
            
        latencies = np.array(latencies)
        
        # Calculate statistics
        metrics = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': float(1000.0 / np.mean(latencies))
        }
        
        return metrics
        
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage in bytes
        """
        total_device_memory = 0
        total_host_memory = 0
        
        for buffer in self.buffers.values():
            total_device_memory += buffer['nbytes']
            total_host_memory += buffer['nbytes']
            
        return {
            'device_memory_bytes': total_device_memory,
            'host_memory_bytes': total_host_memory,
            'total_memory_bytes': total_device_memory + total_host_memory,
            'device_memory_mb': total_device_memory / (1024 * 1024),
            'host_memory_mb': total_host_memory / (1024 * 1024),
            'total_memory_mb': (total_device_memory + total_host_memory) / (1024 * 1024)
        }
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up allocated resources."""
        # Free device memory
        for buffer in self.buffers.values():
            buffer['device'].free()
            
        # Destroy context
        if hasattr(self, 'context'):
            del self.context
            
        # Destroy engine
        if hasattr(self, 'engine'):
            del self.engine
            
        self.logger.info("Resources cleaned up")


def main():
    """Example usage of TensorRT inference engine."""
    import argparse
    from coloredlogs import install as setup_colored_logs
    
    parser = argparse.ArgumentParser(description='TensorRT inference example')
    parser.add_argument('--engine', required=True, help='Path to TensorRT engine')
    parser.add_argument('--input', help='Input image path')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_colored_logs(level=level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create inference engine
        with TensorRTInferenceEngine(
            args.engine,
            max_batch_size=args.batch_size,
            verbose=args.verbose
        ) as engine:
            
            if args.benchmark:
                # Run benchmark
                logger.info("Running benchmark...")
                
                # Create dummy input
                dummy_inputs = {}
                for name, info in engine.bindings_info.items():
                    if info['is_input']:
                        shape = list(info['shape'])
                        if shape[0] == -1:
                            shape[0] = args.batch_size
                            
                        dummy_inputs[name] = np.random.randn(*shape).astype(info['dtype'])
                        
                metrics = engine.benchmark(dummy_inputs)
                
                logger.info("Benchmark Results:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value:.2f}")
                    
            elif args.input:
                # Run inference on image
                logger.info(f"Running inference on {args.input}")
                
                output = engine.infer(engine.preprocess_input(args.input))
                
                logger.info(f"Output shape: {output.shape}")
                logger.info(f"Output min: {output.min():.4f}")
                logger.info(f"Output max: {output.max():.4f}")
                logger.info(f"Output mean: {output.mean():.4f}")
                
            # Show memory usage
            memory = engine.get_memory_usage()
            logger.info(f"Memory usage: {memory['total_memory_mb']:.2f} MB")
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        

if __name__ == '__main__':
    main()