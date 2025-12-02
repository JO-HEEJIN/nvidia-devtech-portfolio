#!/usr/bin/env python3
"""
Asynchronous Triton client for high-throughput inference
"""

import argparse
import asyncio
import aiohttp
import time
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import json


class AsyncTritonClient:
    """
    Asynchronous HTTP client for Triton Inference Server
    """
    
    def __init__(
        self,
        server_url: str = "localhost:8000",
        max_concurrent_requests: int = 100,
        timeout: float = 60.0
    ):
        """
        Initialize async client
        
        Args:
            server_url: Triton server URL
            max_concurrent_requests: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        if not self.server_url.startswith('http'):
            self.server_url = f"http://{self.server_url}"
        
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    async def is_server_ready(self) -> bool:
        """Check if server is ready"""
        try:
            async with self.session.get(f"{self.server_url}/v2/health/ready") as response:
                return response.status == 200
        except:
            return False
    
    async def infer(
        self,
        model_name: str,
        inputs: List[Dict],
        outputs: Optional[List[Dict]] = None,
        model_version: str = "",
        request_id: str = ""
    ) -> Dict:
        """
        Run asynchronous inference
        
        Args:
            model_name: Name of model
            inputs: List of input dictionaries
            outputs: List of output dictionaries (optional)
            model_version: Model version (optional)
            request_id: Request ID for tracking (optional)
        
        Returns:
            Inference response dictionary
        """
        async with self.semaphore:
            url = f"{self.server_url}/v2/models/{model_name}"
            if model_version:
                url += f"/versions/{model_version}"
            url += "/infer"
            
            # Prepare request payload
            payload = {"inputs": inputs}
            if outputs:
                payload["outputs"] = outputs
            if request_id:
                payload["id"] = request_id
            
            # Send request
            start_time = time.perf_counter()
            
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                
            elapsed = (time.perf_counter() - start_time) * 1000
            result['_client_latency_ms'] = elapsed
            
            return result
    
    async def batch_infer(
        self,
        model_name: str,
        batch_inputs: List[List[Dict]],
        model_version: str = ""
    ) -> List[Dict]:
        """
        Run batch of inferences concurrently
        
        Args:
            model_name: Name of model
            batch_inputs: List of input lists
            model_version: Model version (optional)
        
        Returns:
            List of inference results
        """
        tasks = []
        for i, inputs in enumerate(batch_inputs):
            task = self.infer(
                model_name=model_name,
                inputs=inputs,
                model_version=model_version,
                request_id=f"batch_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results


class AsyncModelClient:
    """
    High-level async client for model inference
    """
    
    def __init__(
        self,
        server_url: str = "localhost:8000",
        max_concurrent_requests: int = 100
    ):
        self.server_url = server_url
        self.max_concurrent_requests = max_concurrent_requests
    
    async def classify_images_batch(
        self,
        image_paths: List[str],
        model_name: str = "resnet50_pytorch"
    ) -> List[Dict]:
        """
        Classify batch of images asynchronously
        
        Args:
            image_paths: List of image paths
            model_name: Model name
        
        Returns:
            List of classification results
        """
        async with AsyncTritonClient(
            self.server_url,
            self.max_concurrent_requests
        ) as client:
            # Prepare all inputs
            batch_inputs = []
            for image_path in image_paths:
                image = self._preprocess_resnet50(image_path)
                inputs = [{
                    "name": "input__0",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": image.tolist()
                }]
                batch_inputs.append(inputs)
            
            # Run batch inference
            start_time = time.perf_counter()
            results = await client.batch_infer(model_name, batch_inputs)
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                logits = np.array(result["outputs"][0]["data"]).reshape(1000)
                predicted_class = np.argmax(logits)
                confidence = float(np.max(np.softmax(logits)))
                
                processed_results.append({
                    "image": image_paths[i],
                    "class_id": int(predicted_class),
                    "confidence": confidence,
                    "latency_ms": result["_client_latency_ms"]
                })
            
            # Add summary statistics
            summary = {
                "total_images": len(image_paths),
                "total_time_ms": total_time,
                "avg_latency_ms": total_time / len(image_paths),
                "throughput_qps": len(image_paths) / (total_time / 1000),
                "results": processed_results
            }
            
            return summary
    
    async def detect_objects_stream(
        self,
        video_path: str,
        model_name: str = "yolov8_tensorrt",
        frame_skip: int = 5,
        max_frames: int = 100
    ) -> Dict:
        """
        Stream video frames for object detection
        
        Args:
            video_path: Path to video file
            model_name: Model name
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
        
        Returns:
            Detection results summary
        """
        import cv2
        
        async with AsyncTritonClient(
            self.server_url,
            self.max_concurrent_requests
        ) as client:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frames_to_process = []
            frame_indices = []
            frame_count = 0
            
            # Extract frames
            while cap.isOpened() and len(frames_to_process) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    frames_to_process.append(frame)
                    frame_indices.append(frame_count)
                
                frame_count += 1
            
            cap.release()
            
            # Prepare inputs
            batch_inputs = []
            for frame in frames_to_process:
                processed = self._preprocess_yolov8_frame(frame)
                inputs = [{
                    "name": "images",
                    "shape": [1, 3, 640, 640],
                    "datatype": "FP32",
                    "data": processed.tolist()
                }]
                batch_inputs.append(inputs)
            
            # Run batch inference
            start_time = time.perf_counter()
            results = await client.batch_infer(model_name, batch_inputs)
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Process results
            all_detections = []
            for i, result in enumerate(results):
                detections = np.array(result["outputs"][0]["data"])
                num_detections = len(detections) // 84
                detections = detections.reshape(num_detections, 84)
                
                # Filter by confidence
                confident_detections = detections[detections[:, 4] > 0.5]
                
                all_detections.append({
                    "frame_index": frame_indices[i],
                    "timestamp": frame_indices[i] / fps,
                    "num_detections": len(confident_detections)
                })
            
            # Summary
            summary = {
                "video_path": video_path,
                "total_frames_processed": len(frames_to_process),
                "total_time_ms": total_time,
                "avg_latency_ms": total_time / len(frames_to_process),
                "throughput_fps": len(frames_to_process) / (total_time / 1000),
                "detections": all_detections
            }
            
            return summary
    
    def _preprocess_resnet50(self, image_path: str) -> np.ndarray:
        """Preprocess image for ResNet50"""
        import cv2
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        # Center crop
        h, w = image.shape[:2]
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        image = image[start_h:start_h+224, start_w:start_w+224]
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    def _preprocess_yolov8_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess video frame for YOLOv8"""
        import cv2
        
        # Resize frame
        frame = cv2.resize(frame, (640, 640))
        
        # Convert BGR to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to CHW format
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        
        return frame.astype(np.float32)


async def stress_test(
    server_url: str = "localhost:8000",
    model_name: str = "resnet50_pytorch",
    num_requests: int = 1000,
    concurrent_requests: int = 50
):
    """
    Stress test Triton server with concurrent requests
    """
    print(f"Starting stress test...")
    print(f"  Server: {server_url}")
    print(f"  Model: {model_name}")
    print(f"  Total requests: {num_requests}")
    print(f"  Concurrent requests: {concurrent_requests}")
    
    # Create dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    inputs = [{
        "name": "input__0",
        "shape": [1, 3, 224, 224],
        "datatype": "FP32",
        "data": dummy_input.tolist()
    }]
    
    async with AsyncTritonClient(
        server_url,
        concurrent_requests
    ) as client:
        # Check server is ready
        if not await client.is_server_ready():
            print("Error: Server is not ready")
            return
        
        # Prepare requests
        tasks = []
        for i in range(num_requests):
            task = client.infer(
                model_name=model_name,
                inputs=inputs,
                request_id=f"stress_{i}"
            )
            tasks.append(task)
        
        # Run all requests
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = num_requests - successful
        
        latencies = [
            r["_client_latency_ms"] 
            for r in results 
            if not isinstance(r, Exception)
        ]
        
        if latencies:
            print(f"\nResults:")
            print(f"  Successful: {successful}/{num_requests}")
            print(f"  Failed: {failed}/{num_requests}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {successful/total_time:.2f} QPS")
            print(f"\nLatency Statistics:")
            print(f"  Mean: {np.mean(latencies):.2f} ms")
            print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
            print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
            print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
            print(f"  Max: {np.max(latencies):.2f} ms")


async def main():
    parser = argparse.ArgumentParser(description='Async Triton Client')
    parser.add_argument('--server', type=str, default='localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['batch', 'stream', 'stress'],
                        help='Operation mode')
    parser.add_argument('--model', type=str, default='resnet50_pytorch',
                        help='Model name')
    parser.add_argument('--input', type=str, nargs='+',
                        help='Input files')
    parser.add_argument('--concurrent', type=int, default=50,
                        help='Max concurrent requests')
    parser.add_argument('--requests', type=int, default=1000,
                        help='Total requests for stress test')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        if not args.input:
            print("Error: --input required for batch mode")
            return
        
        client = AsyncModelClient(args.server, args.concurrent)
        result = await client.classify_images_batch(args.input, args.model)
        
        print(f"Batch Classification Results:")
        print(f"  Total images: {result['total_images']}")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_qps']:.2f} QPS")
        print(f"\nFirst 5 results:")
        for r in result['results'][:5]:
            print(f"  {r['image']}: Class {r['class_id']} ({r['confidence']:.2%})")
    
    elif args.mode == 'stream':
        if not args.input or len(args.input) != 1:
            print("Error: Single video file required for stream mode")
            return
        
        client = AsyncModelClient(args.server, args.concurrent)
        result = await client.detect_objects_stream(args.input[0])
        
        print(f"Video Stream Detection Results:")
        print(f"  Video: {result['video_path']}")
        print(f"  Frames processed: {result['total_frames_processed']}")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Throughput: {result['throughput_fps']:.2f} FPS")
    
    elif args.mode == 'stress':
        await stress_test(
            args.server,
            args.model,
            args.requests,
            args.concurrent
        )


if __name__ == '__main__':
    asyncio.run(main())