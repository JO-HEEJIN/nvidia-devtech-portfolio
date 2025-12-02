"""
PyTorch inference baseline for YOLOv8
"""

import time
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional

from preprocessing import preprocess_image, preprocess_batch
from postprocessing import postprocess_predictions, scale_boxes_to_original


class PyTorchInference:
    """
    PyTorch inference wrapper for YOLOv8
    """
    
    def __init__(
        self,
        model_path: str = 'yolov8s.pt',
        device: str = 'cuda',
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize PyTorch inference
        
        Args:
            model_path: Path to YOLOv8 model
            device: Device for inference (cuda/cpu)
            input_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
        """
        self.device = device
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"Loading PyTorch model from {model_path}")
        self.model = YOLO(model_path)
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.model.to('cpu')
            print("  Using CPU")
        
        # Warmup
        self.warmup()
    
    def warmup(self, iterations: int = 3):
        """
        Warmup model with dummy input
        """
        print("Warming up model...")
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        
        for _ in range(iterations):
            _ = self.model(dummy, verbose=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
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
        
        # Run inference
        start_time = time.perf_counter()
        
        # Convert to torch tensor
        if self.device == 'cuda':
            input_tensor = torch.from_numpy(tensor).cuda()
        else:
            input_tensor = torch.from_numpy(tensor)
        
        # Inference
        with torch.no_grad():
            results = self.model(input_tensor, verbose=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Extract detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Get box coordinates, confidence, and class
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            # Combine into detections array
            detections = np.concatenate([
                xyxy,
                conf[:, np.newaxis],
                cls[:, np.newaxis]
            ], axis=1)
            
            # Scale boxes to original image
            detections[:, :4] = scale_boxes_to_original(
                detections[:, :4].copy(), ratio, padding
            )
        else:
            detections = np.empty((0, 6))
        
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
        # Preprocess batch
        batch_tensor, originals, ratios, paddings = preprocess_batch(
            images, self.input_size
        )
        
        # Convert to torch
        if self.device == 'cuda':
            input_tensor = torch.from_numpy(batch_tensor).cuda()
        else:
            input_tensor = torch.from_numpy(batch_tensor)
        
        # Run inference
        with torch.no_grad():
            results = self.model(input_tensor, verbose=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Process results
        batch_detections = []
        for i, result in enumerate(results):
            if result.boxes is not None:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                detections = np.concatenate([
                    xyxy,
                    conf[:, np.newaxis],
                    cls[:, np.newaxis]
                ], axis=1)
                
                # Scale to original
                detections[:, :4] = scale_boxes_to_original(
                    detections[:, :4].copy(), ratios[i], paddings[i]
                )
            else:
                detections = np.empty((0, 6))
            
            batch_detections.append(detections)
        
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
        
        if self.device == 'cuda':
            input_tensor = torch.from_numpy(tensor).cuda()
        else:
            input_tensor = torch.from_numpy(tensor)
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(input_tensor, verbose=False)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(input_tensor, verbose=False)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
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


def measure_pytorch_fps(
    model_path: str = 'yolov8s.pt',
    input_size: int = 640,
    batch_size: int = 1,
    iterations: int = 100
) -> float:
    """
    Measure PyTorch FPS
    
    Args:
        model_path: Path to model
        input_size: Input image size
        batch_size: Batch size
        iterations: Number of iterations
    
    Returns:
        Average FPS
    """
    # Create inference engine
    engine = PyTorchInference(model_path, input_size=(input_size, input_size))
    
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