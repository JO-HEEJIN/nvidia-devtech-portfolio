#!/usr/bin/env python3
"""
Triton HTTP/REST client for inference
"""

import argparse
import json
import time
import numpy as np
import requests
from typing import Dict, List, Any
import cv2
from PIL import Image


class TritonHTTPClient:
    """
    HTTP client for Triton Inference Server
    """
    
    def __init__(self, server_url: str = "localhost:8000", timeout: float = 60.0):
        """
        Initialize HTTP client
        
        Args:
            server_url: Triton server URL
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        if not self.server_url.startswith('http'):
            self.server_url = f"http://{self.server_url}"
        
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def is_server_live(self) -> bool:
        """Check if server is live"""
        try:
            response = self.session.get(f"{self.server_url}/v2/health/live", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def is_server_ready(self) -> bool:
        """Check if server is ready"""
        try:
            response = self.session.get(f"{self.server_url}/v2/health/ready", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """Check if model is ready"""
        try:
            url = f"{self.server_url}/v2/models/{model_name}"
            if model_version:
                url += f"/versions/{model_version}"
            url += "/ready"
            
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict:
        """Get model metadata"""
        url = f"{self.server_url}/v2/models/{model_name}"
        if model_version:
            url += f"/versions/{model_version}"
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_server_metadata(self) -> Dict:
        """Get server metadata"""
        response = self.session.get(f"{self.server_url}/v2", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def infer(
        self,
        model_name: str,
        inputs: List[Dict],
        outputs: List[Dict] = None,
        model_version: str = "",
        request_id: str = ""
    ) -> Dict:
        """
        Run inference
        
        Args:
            model_name: Name of model
            inputs: List of input dictionaries
            outputs: List of output dictionaries (optional)
            model_version: Model version (optional)
            request_id: Request ID for tracking (optional)
        
        Returns:
            Inference response dictionary
        """
        url = f"{self.server_url}/v2/models/{model_name}"
        if model_version:
            url += f"/versions/{model_version}"
        url += "/infer"
        
        # Prepare request payload
        payload = {
            "inputs": inputs
        }
        
        if outputs:
            payload["outputs"] = outputs
        
        if request_id:
            payload["id"] = request_id
        
        # Send request
        start_time = time.perf_counter()
        response = self.session.post(url, json=payload, timeout=self.timeout)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        response.raise_for_status()
        result = response.json()
        result['_client_latency_ms'] = elapsed
        
        return result


class ModelClient:
    """
    High-level client for specific model types
    """
    
    def __init__(self, server_url: str = "localhost:8000"):
        self.client = TritonHTTPClient(server_url)
    
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
        
        # Prepare input
        inputs = [{
            "name": "input__0",
            "shape": [1, 3, 224, 224],
            "datatype": "FP32",
            "data": image.tolist()
        }]
        
        # Run inference
        result = self.client.infer("resnet50_pytorch", inputs)
        
        # Process output
        logits = np.array(result["outputs"][0]["data"]).reshape(1000)
        predicted_class = np.argmax(logits)
        confidence = float(np.max(np.softmax(logits)))
        
        return {
            "class_id": int(predicted_class),
            "confidence": confidence,
            "latency_ms": result["_client_latency_ms"]
        }
    
    def detect_objects_yolov8(self, image_path: str) -> Dict:
        """
        Detect objects with YOLOv8
        
        Args:
            image_path: Path to image file
        
        Returns:
            Detection results
        """
        # Load and preprocess image
        image = self._preprocess_yolov8(image_path)
        
        # Prepare input
        inputs = [{
            "name": "images",
            "shape": [1, 3, 640, 640],
            "datatype": "FP32",
            "data": image.tolist()
        }]
        
        # Run inference
        result = self.client.infer("yolov8_tensorrt", inputs)
        
        # Process output
        detections = np.array(result["outputs"][0]["data"])
        num_detections = len(detections) // 84
        detections = detections.reshape(num_detections, 84)
        
        # Filter by confidence
        confident_detections = detections[detections[:, 4] > 0.5]
        
        return {
            "num_detections": len(confident_detections),
            "detections": confident_detections.tolist(),
            "latency_ms": result["_client_latency_ms"]
        }
    
    def classify_text_bert(self, text: str) -> Dict:
        """
        Classify text with BERT
        
        Args:
            text: Input text
        
        Returns:
            Classification result
        """
        # Preprocess text (simplified tokenization)
        input_ids, attention_mask = self._preprocess_bert(text)
        
        # Prepare inputs
        inputs = [
            {
                "name": "input_ids",
                "shape": [1, 128],
                "datatype": "INT64",
                "data": input_ids.tolist()
            },
            {
                "name": "attention_mask",
                "shape": [1, 128],
                "datatype": "INT64",
                "data": attention_mask.tolist()
            }
        ]
        
        # Run inference
        result = self.client.infer("bert_onnx", inputs)
        
        # Process output
        logits = np.array(result["outputs"][0]["data"]).reshape(2)
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
            "latency_ms": result["_client_latency_ms"]
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
        
        return image
    
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
        
        return image
    
    def _preprocess_bert(self, text: str) -> tuple:
        """Simplified BERT preprocessing (requires transformers for full implementation)"""
        # This is a simplified version - in practice, use transformers tokenizer
        # For demo purposes, create dummy tokens
        tokens = text.lower().split()[:126]  # Reserve 2 for special tokens
        
        # Create dummy input_ids and attention_mask
        input_ids = np.zeros(128, dtype=np.int64)
        attention_mask = np.zeros(128, dtype=np.int64)
        
        # Add [CLS] token
        input_ids[0] = 101  # CLS token
        attention_mask[0] = 1
        
        # Add tokens (simplified mapping)
        for i, token in enumerate(tokens):
            if i < 126:
                input_ids[i + 1] = hash(token) % 30000 + 1000  # Dummy token ID
                attention_mask[i + 1] = 1
        
        # Add [SEP] token
        if len(tokens) < 126:
            input_ids[len(tokens) + 1] = 102  # SEP token
            attention_mask[len(tokens) + 1] = 1
        
        return input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description='Triton HTTP Client')
    parser.add_argument('--server', type=str, default='localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'yolov8', 'bert'],
                        help='Model to test')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file (image for resnet50/yolov8, text for bert)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Create client
    client = ModelClient(args.server)
    
    # Check server status
    if not client.client.is_server_live():
        print(f"Error: Server {args.server} is not live")
        return
    
    if not client.client.is_server_ready():
        print(f"Error: Server {args.server} is not ready")
        return
    
    # Run inference
    print(f"Running inference with {args.model} model...")
    
    try:
        if args.model == 'resnet50':
            model_name = 'resnet50_pytorch'
            if not client.client.is_model_ready(model_name):
                print(f"Error: Model {model_name} is not ready")
                return
            
            result = client.classify_image_resnet50(args.input)
            print(f"Classification Result:")
            print(f"  Class ID: {result['class_id']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
        
        elif args.model == 'yolov8':
            model_name = 'yolov8_tensorrt'
            if not client.client.is_model_ready(model_name):
                print(f"Error: Model {model_name} is not ready")
                return
            
            result = client.detect_objects_yolov8(args.input)
            print(f"Detection Result:")
            print(f"  Number of detections: {result['num_detections']}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
        
        elif args.model == 'bert':
            model_name = 'bert_onnx'
            if not client.client.is_model_ready(model_name):
                print(f"Error: Model {model_name} is not ready")
                return
            
            # For BERT, treat input as text
            text = args.input if len(args.input) > 20 else "This is a test sentence."
            result = client.classify_text_bert(text)
            print(f"Text Classification Result:")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Latency: {result['latency_ms']:.2f} ms")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()