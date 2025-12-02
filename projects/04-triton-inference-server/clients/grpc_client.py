#!/usr/bin/env python3
"""
Triton gRPC client for inference
"""

import argparse
import time
import numpy as np
from typing import List, Dict, Any, Optional
import grpc
import cv2

# Import Triton client library
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError:
    print("tritonclient not installed. Install with: pip install tritonclient[grpc]")
    exit(1)


class TritonGRPCClient:
    """
    gRPC client for Triton Inference Server
    """
    
    def __init__(
        self,
        server_url: str = "localhost:8001",
        timeout: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize gRPC client
        
        Args:
            server_url: Triton server URL
            timeout: Request timeout in seconds
            verbose: Verbose logging
        """
        self.server_url = server_url
        self.timeout = timeout
        self.verbose = verbose
        
        # Create client
        try:
            self.client = grpcclient.InferenceServerClient(
                url=server_url,
                verbose=verbose
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create gRPC client: {e}")
    
    def is_server_live(self) -> bool:
        """Check if server is live"""
        try:
            return self.client.is_server_live()
        except:
            return False
    
    def is_server_ready(self) -> bool:
        """Check if server is ready"""
        try:
            return self.client.is_server_ready()
        except:
            return False
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """Check if model is ready"""
        try:
            return self.client.is_model_ready(model_name, model_version)
        except:
            return False
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict:
        """Get model metadata"""
        try:
            metadata = self.client.get_model_metadata(
                model_name=model_name,
                model_version=model_version
            )
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape
                    } for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape
                    } for out in metadata.outputs
                ]
            }
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get model metadata: {e}")
    
    def get_server_metadata(self) -> Dict:
        """Get server metadata"""
        try:
            metadata = self.client.get_server_metadata()
            return {
                "name": metadata.name,
                "version": metadata.version,
                "extensions": metadata.extensions
            }
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to get server metadata: {e}")
    
    def infer(
        self,
        model_name: str,
        inputs: List[grpcclient.InferInput],
        outputs: Optional[List[grpcclient.InferRequestedOutput]] = None,
        model_version: str = "",
        request_id: str = "",
        headers: Optional[Dict] = None
    ) -> grpcclient.InferResult:
        """
        Run inference
        
        Args:
            model_name: Name of model
            inputs: List of InferInput objects
            outputs: List of InferRequestedOutput objects (optional)
            model_version: Model version (optional)
            request_id: Request ID for tracking (optional)
            headers: Additional headers (optional)
        
        Returns:
            InferResult object
        """
        try:
            start_time = time.perf_counter()
            
            result = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                model_version=model_version,
                request_id=request_id,
                headers=headers
            )
            
            elapsed = (time.perf_counter() - start_time) * 1000
            result._client_latency_ms = elapsed
            
            return result
            
        except InferenceServerException as e:
            raise RuntimeError(f"Inference failed: {e}")


class ModelClient:
    """
    High-level client for specific model types
    """
    
    def __init__(self, server_url: str = "localhost:8001", verbose: bool = False):
        self.client = TritonGRPCClient(server_url, verbose=verbose)
    
    def classify_image_resnet50(self, image_path: str) -> Dict:
        """
        Classify image with ResNet50
        
        Args:
            image_path: Path to image file
        
        Returns:
            Classification result
        """
        # Load and preprocess image
        image = self._preprocess_resnet50(image_path)
        
        # Create input
        inputs = []
        input_tensor = grpcclient.InferInput("input__0", [1, 3, 224, 224], "FP32")
        input_tensor.set_data_from_numpy(image)
        inputs.append(input_tensor)
        
        # Create output request
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("output__0"))
        
        # Run inference
        result = self.client.infer("resnet50_pytorch", inputs, outputs)
        
        # Process output
        logits = result.as_numpy("output__0").reshape(1000)
        predicted_class = np.argmax(logits)
        confidence = float(np.max(np.softmax(logits)))
        
        return {
            "class_id": int(predicted_class),
            "confidence": confidence,
            "latency_ms": result._client_latency_ms
        }
    
    def detect_objects_yolov8(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect objects with YOLOv8
        
        Args:
            image_path: Path to image file
            confidence_threshold: Confidence threshold for detections
        
        Returns:
            Detection results
        """
        # Load and preprocess image
        image = self._preprocess_yolov8(image_path)
        
        # Create input
        inputs = []
        input_tensor = grpcclient.InferInput("images", [1, 3, 640, 640], "FP32")
        input_tensor.set_data_from_numpy(image)
        inputs.append(input_tensor)
        
        # Create output request
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("output0"))
        
        # Run inference
        result = self.client.infer("yolov8_tensorrt", inputs, outputs)
        
        # Process output
        detections = result.as_numpy("output0")
        num_detections = detections.shape[0] // 84
        detections = detections.reshape(num_detections, 84)
        
        # Filter by confidence
        confident_detections = detections[detections[:, 4] > confidence_threshold]
        
        # Parse detections
        parsed_detections = []
        for det in confident_detections:
            x, y, w, h = det[:4]
            confidence = det[4]
            class_scores = det[5:]
            class_id = np.argmax(class_scores)
            
            parsed_detections.append({
                "bbox": [float(x), float(y), float(w), float(h)],
                "confidence": float(confidence),
                "class_id": int(class_id)
            })
        
        return {
            "num_detections": len(parsed_detections),
            "detections": parsed_detections,
            "latency_ms": result._client_latency_ms
        }
    
    def classify_text_bert(self, text: str) -> Dict:
        """
        Classify text with BERT
        
        Args:
            text: Input text
        
        Returns:
            Classification result
        """
        # Preprocess text
        input_ids, attention_mask = self._preprocess_bert(text)
        
        # Create inputs
        inputs = []
        
        input_ids_tensor = grpcclient.InferInput("input_ids", [1, 128], "INT64")
        input_ids_tensor.set_data_from_numpy(input_ids)
        inputs.append(input_ids_tensor)
        
        attention_mask_tensor = grpcclient.InferInput("attention_mask", [1, 128], "INT64")
        attention_mask_tensor.set_data_from_numpy(attention_mask)
        inputs.append(attention_mask_tensor)
        
        # Create output request
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("logits"))
        
        # Run inference
        result = self.client.infer("bert_onnx", inputs, outputs)
        
        # Process output
        logits = result.as_numpy("logits").reshape(2)
        probabilities = np.softmax(logits)
        predicted_class = np.argmax(probabilities)
        
        labels = ["negative", "positive"]
        
        return {
            "label": labels[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            },
            "latency_ms": result._client_latency_ms
        }
    
    def _preprocess_resnet50(self, image_path: str) -> np.ndarray:
        """Preprocess image for ResNet50"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and center crop
        image = cv2.resize(image, (256, 256))
        h, w = image.shape[:2]
        start_h = (h - 224) // 2
        start_w = (w - 224) // 2
        image = image[start_h:start_h+224, start_w:start_w+224]
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    def _preprocess_yolov8(self, image_path: str) -> np.ndarray:
        """Preprocess image for YOLOv8"""
        # Load and resize image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    def _preprocess_bert(self, text: str) -> tuple:
        """Simplified BERT preprocessing"""
        # This is a simplified version - in practice, use transformers tokenizer
        tokens = text.lower().split()[:126]
        
        # Create dummy input_ids and attention_mask
        input_ids = np.zeros((1, 128), dtype=np.int64)
        attention_mask = np.zeros((1, 128), dtype=np.int64)
        
        # Add [CLS] token
        input_ids[0, 0] = 101
        attention_mask[0, 0] = 1
        
        # Add tokens
        for i, token in enumerate(tokens):
            if i < 126:
                input_ids[0, i + 1] = hash(token) % 30000 + 1000
                attention_mask[0, i + 1] = 1
        
        # Add [SEP] token
        if len(tokens) < 126:
            input_ids[0, len(tokens) + 1] = 102
            attention_mask[0, len(tokens) + 1] = 1
        
        return input_ids, attention_mask


def benchmark_grpc_vs_http(
    server_url_grpc: str = "localhost:8001",
    server_url_http: str = "localhost:8000",
    model_name: str = "resnet50_pytorch",
    iterations: int = 100
):
    """
    Benchmark gRPC vs HTTP performance
    """
    import sys
    sys.path.append('.')
    from http_client import TritonHTTPClient
    
    # Prepare dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # gRPC benchmark
    grpc_client = TritonGRPCClient(server_url_grpc)
    
    grpc_times = []
    for _ in range(iterations):
        inputs = []
        input_tensor = grpcclient.InferInput("input__0", [1, 3, 224, 224], "FP32")
        input_tensor.set_data_from_numpy(dummy_input)
        inputs.append(input_tensor)
        
        start = time.perf_counter()
        result = grpc_client.infer(model_name, inputs)
        grpc_times.append((time.perf_counter() - start) * 1000)
    
    # HTTP benchmark
    http_client = TritonHTTPClient(server_url_http)
    
    http_times = []
    for _ in range(iterations):
        inputs = [{
            "name": "input__0",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": dummy_input.tolist()
        }]
        
        start = time.perf_counter()
        result = http_client.infer(model_name, inputs)
        http_times.append((time.perf_counter() - start) * 1000)
    
    # Print results
    print(f"\nBenchmark Results ({iterations} iterations):")
    print("-" * 40)
    print(f"gRPC Latency:")
    print(f"  Mean: {np.mean(grpc_times):.2f} ms")
    print(f"  P50: {np.percentile(grpc_times, 50):.2f} ms")
    print(f"  P95: {np.percentile(grpc_times, 95):.2f} ms")
    print(f"  P99: {np.percentile(grpc_times, 99):.2f} ms")
    print(f"\nHTTP Latency:")
    print(f"  Mean: {np.mean(http_times):.2f} ms")
    print(f"  P50: {np.percentile(http_times, 50):.2f} ms")
    print(f"  P95: {np.percentile(http_times, 95):.2f} ms")
    print(f"  P99: {np.percentile(http_times, 99):.2f} ms")
    print(f"\nPerformance Gain:")
    print(f"  gRPC is {np.mean(http_times) / np.mean(grpc_times):.2f}x faster")


def main():
    parser = argparse.ArgumentParser(description='Triton gRPC Client')
    parser.add_argument('--server', type=str, default='localhost:8001',
                        help='Triton server URL')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'yolov8', 'bert'],
                        help='Model to test')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (image for resnet50/yolov8, text for bert)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark comparison')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_grpc_vs_http()
        return
    
    # Create client
    client = ModelClient(args.server, verbose=args.verbose)
    
    # Check server status
    if not client.client.is_server_live():
        print(f"Error: Server {args.server} is not live")
        return
    
    if not client.client.is_server_ready():
        print(f"Error: Server {args.server} is not ready")
        return
    
    # Run inference
    print(f"Running gRPC inference with {args.model} model...")
    
    try:
        if args.model == 'resnet50':
            result = client.classify_image_resnet50(args.input)
            print(f"Classification Result:")
            print(f"  Class ID: {result['class_id']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
        
        elif args.model == 'yolov8':
            result = client.detect_objects_yolov8(args.input)
            print(f"Detection Result:")
            print(f"  Number of detections: {result['num_detections']}")
            if result['num_detections'] > 0:
                print(f"  First detection: {result['detections'][0]}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
        
        elif args.model == 'bert':
            text = args.input if len(args.input) > 20 else "This is a great product!"
            result = client.classify_text_bert(text)
            print(f"Text Classification Result:")
            print(f"  Text: '{text[:50]}...'")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()