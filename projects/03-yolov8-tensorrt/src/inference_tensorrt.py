"""
TensorRT optimized inference for YOLOv8
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    warnings.warn("TensorRT or PyCUDA not found. TensorRT inference will not be available.")

from preprocessing import preprocess_image, preprocess_batch
from postprocessing import postprocess_predictions, scale_boxes_to_original


class TensorRTInference:
    """
    TensorRT inference engine for YOLOv8
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_cuda_graph: bool = False
    ):
        """
        Initialize TensorRT inference
        
        Args:
            engine_path: Path to TensorRT engine
            input_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            use_cuda_graph: Use CUDA graphs for reduced overhead
        """
        self.engine_path = Path(engine_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_cuda_graph = use_cuda_graph
        
        # Load engine
        self.engine, self.context = self.load_engine()
        
        # Setup bindings
        self.setup_bindings()
        
        # CUDA stream for async execution
        self.stream = cuda.Stream()
        
        # Memory pool for efficient allocation
        self.memory_pool = {}
        
        # Warmup
        self.warmup()
    
    def load_engine(self):
        """
        Load TensorRT engine from file
        """
        print(f"Loading TensorRT engine from {self.engine_path}")
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        # Create runtime
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        # Load engine
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to load engine")
        
        # Create execution context
        context = engine.create_execution_context()
        
        print(f"  Engine loaded successfully")
        print(f"  Max batch size: {engine.max_batch_size}")
        
        return engine, context
    
    def setup_bindings(self):
        """
        Setup input/output bindings
        """
        self.bindings = []
        self.binding_shapes = {}
        self.binding_names = {}
        self.binding_dtypes = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            
            # Calculate size
            size = trt.volume(shape)
            if dtype == trt.DataType.FLOAT:
                np_dtype = np.float32
            elif dtype == trt.DataType.HALF:
                np_dtype = np.float16
            elif dtype == trt.DataType.INT8:
                np_dtype = np.int8
            else:
                np_dtype = np.float32
            
            # Allocate memory
            if self.engine.binding_is_input(i):
                self.input_binding = i
                self.input_shape = shape
                self.input_dtype = np_dtype
                self.input_size_bytes = size * np_dtype().itemsize
                
                # Allocate host and device memory
                self.host_input = cuda.pagelocked_empty(size, np_dtype)
                self.device_input = cuda.mem_alloc(self.input_size_bytes)
                self.bindings.append(int(self.device_input))
                
                print(f"  Input: {name} - {shape} ({dtype})")
            else:
                self.output_binding = i
                self.output_shape = shape
                self.output_dtype = np_dtype
                self.output_size_bytes = size * np_dtype().itemsize
                
                # Allocate host and device memory
                self.host_output = cuda.pagelocked_empty(size, np_dtype)
                self.device_output = cuda.mem_alloc(self.output_size_bytes)
                self.bindings.append(int(self.device_output))
                
                print(f"  Output: {name} - {shape} ({dtype})")
    
    def warmup(self, iterations: int = 3):
        """
        Warmup engine with dummy input
        """
        print("Warming up engine...")
        dummy = np.random.randn(*self.input_shape).astype(self.input_dtype)
        
        for _ in range(iterations):
            self._execute(dummy)
        
        self.stream.synchronize()
    
    def _execute(self, input_data: np.ndarray) -> np.ndarray:
        """
        Execute inference on device
        
        Args:
            input_data: Preprocessed input tensor
        
        Returns:
            Raw output tensor
        """
        # Copy input to host memory
        np.copyto(self.host_input, input_data.ravel())
        
        # Transfer to device
        cuda.memcpy_htod_async(
            self.device_input,
            self.host_input,
            self.stream
        )
        
        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer output to host
        cuda.memcpy_dtoh_async(
            self.host_output,
            self.device_output,
            self.stream
        )
        
        # Synchronize
        self.stream.synchronize()
        
        # Reshape output
        output = self.host_output.reshape(self.output_shape)
        
        return output
    
    def infer(
        self,
        image: np.ndarray,
        return_preprocessed: bool = False
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Run inference on single image
        
        Args:
            image: Input image (BGR)
            return_preprocessed: Return preprocessing info
        
        Returns:
            Detections array [N, 6] with [x1, y1, x2, y2, conf, class]
            Optional preprocessing info dict
        """
        # Preprocess
        tensor, original, ratio, padding = preprocess_image(
            image, self.input_size, normalize=True, bgr_to_rgb=True
        )
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Execute
        output = self._execute(tensor)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Postprocess
        detections = postprocess_predictions(
            output,
            self.conf_threshold,
            self.iou_threshold
        )[0]
        
        # Scale boxes to original image
        if len(detections) > 0:
            detections[:, :4] = scale_boxes_to_original(
                detections[:, :4].copy(), ratio, padding
            )
        
        if return_preprocessed:
            info = {
                'inference_time': inference_time,
                'ratio': ratio,
                'padding': padding,
                'input_shape': tensor.shape
            }
            return detections, info
        
        return detections, None
    
    def infer_batch(
        self,
        images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Run inference on batch of images
        
        Args:
            images: List of input images
        
        Returns:
            List of detections per image
        """
        # Check batch size
        batch_size = len(images)
        if batch_size > self.engine.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds engine max {self.engine.max_batch_size}")
        
        # Preprocess batch
        batch_tensor, originals, ratios, paddings = preprocess_batch(
            images, self.input_size
        )
        
        # Execute
        output = self._execute(batch_tensor)
        
        # Postprocess
        batch_detections = postprocess_predictions(
            output,
            self.conf_threshold,
            self.iou_threshold
        )
        
        # Scale boxes to original
        for i, detections in enumerate(batch_detections):
            if len(detections) > 0:
                detections[:, :4] = scale_boxes_to_original(
                    detections[:, :4].copy(), ratios[i], paddings[i]
                )
        
        return batch_detections
    
    def benchmark(
        self,
        image: np.ndarray,
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> dict:
        """
        Benchmark inference performance
        
        Args:
            image: Test image
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
        
        Returns:
            Performance statistics dict
        """
        # Preprocess once
        tensor, _, _, _ = preprocess_image(image, self.input_size)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self._execute(tensor)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self._execute(tensor)
            times.append((time.perf_counter() - start) * 1000)
        
        times = np.array(times)
        
        return {
            'mean_latency': np.mean(times),
            'std_latency': np.std(times),
            'min_latency': np.min(times),
            'max_latency': np.max(times),
            'p50_latency': np.percentile(times, 50),
            'p95_latency': np.percentile(times, 95),
            'p99_latency': np.percentile(times, 99),
            'fps': 1000 / np.mean(times)
        }
    
    def __del__(self):
        """
        Cleanup resources
        """
        # Clean up CUDA resources
        if hasattr(self, 'stream'):
            del self.stream
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


def measure_tensorrt_fps(
    engine_path: str,
    input_size: int = 640,
    batch_size: int = 1,
    iterations: int = 100
) -> float:
    """
    Measure TensorRT FPS
    
    Args:
        engine_path: Path to engine
        input_size: Input image size
        batch_size: Batch size
        iterations: Number of iterations
    
    Returns:
        Average FPS
    """
    # Create inference engine
    engine = TensorRTInference(engine_path, input_size=(input_size, input_size))
    
    # Create dummy input
    dummy = np.random.randint(0, 255, (batch_size, input_size, input_size, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        if batch_size == 1:
            _ = engine.infer(dummy[0])
        else:
            _ = engine.infer_batch(list(dummy))
    
    # Measure
    start = time.perf_counter()
    
    for _ in range(iterations):
        if batch_size == 1:
            _ = engine.infer(dummy[0])
        else:
            _ = engine.infer_batch(list(dummy))
    
    elapsed = time.perf_counter() - start
    fps = (iterations * batch_size) / elapsed
    
    return fps