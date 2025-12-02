#!/usr/bin/env python3
"""
Triton gRPC client with async inference support
Uses tritonclient.grpc for binary protocol communication
"""

import numpy as np
import sys
import time
import argparse
import queue
from PIL import Image
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def preprocess_image(image_path):
    """Preprocess image for ResNet50 inference"""
    # Load and resize image
    img = Image.open(image_path)
    img = img.resize((256, 256))
    
    # Center crop to 224x224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)


def callback(user_data, result, error):
    """Callback function for async inference"""
    if error:
        user_data.put(error)
    else:
        user_data.put(result)


def run_sync_inference(server_url, model_name, image_path):
    """Run synchronous inference using gRPC client"""
    
    # Create client
    triton_client = grpcclient.InferenceServerClient(
        url=server_url,
        verbose=False
    )
    
    # Check if server is live
    if not triton_client.is_server_live():
        print("Server is not live")
        sys.exit(1)
    
    # Check if server is ready
    if not triton_client.is_server_ready():
        print("Server is not ready")
        sys.exit(1)
    
    # Check if model is ready
    if not triton_client.is_model_ready(model_name):
        print(f"Model {model_name} is not ready")
        sys.exit(1)
    
    # Get model metadata
    model_metadata = triton_client.get_model_metadata(model_name)
    model_config = triton_client.get_model_config(model_name)
    
    print(f"Model: {model_name}")
    print(f"Version: {model_metadata.versions}")
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Create input object
    inputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Create output object
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output__0'))
    
    # Measure inference latency
    start_time = time.perf_counter()
    
    # Send request
    try:
        response = triton_client.infer(
            model_name,
            inputs,
            outputs=outputs
        )
    except InferenceServerException as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Parse response
    output_data = response.as_numpy('output__0')
    
    # Get top 5 predictions
    top5_indices = np.argsort(output_data[0])[-5:][::-1]
    
    print(f"\nSynchronous Inference Results:")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Output shape: {output_data.shape}")
    print(f"Top 5 predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"  {i+1}. Class {idx}: {output_data[0][idx]:.4f}")
    
    return output_data, latency_ms


def run_async_inference(server_url, model_name, image_path):
    """Run asynchronous inference using gRPC client"""
    
    # Create client
    triton_client = grpcclient.InferenceServerClient(
        url=server_url,
        verbose=False
    )
    
    print(f"\nAsynchronous Inference:")
    print(f"Model: {model_name}")
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Create input object
    inputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Create output object
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output__0'))
    
    # Create queue for async results
    user_data = queue.Queue()
    
    # Measure inference latency
    start_time = time.perf_counter()
    
    # Send async request
    try:
        triton_client.async_infer(
            model_name,
            inputs,
            callback=lambda result, error: callback(user_data, result, error),
            outputs=outputs
        )
    except InferenceServerException as e:
        print(f"Async inference failed: {e}")
        sys.exit(1)
    
    # Wait for result
    data_item = user_data.get()
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    if isinstance(data_item, InferenceServerException):
        print(f"Async inference error: {data_item}")
        sys.exit(1)
    
    # Parse response
    output_data = data_item.as_numpy('output__0')
    
    # Get top 5 predictions
    top5_indices = np.argsort(output_data[0])[-5:][::-1]
    
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Top 5 predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"  {i+1}. Class {idx}: {output_data[0][idx]:.4f}")
    
    return output_data, latency_ms


def run_streaming_inference(server_url, model_name, image_paths):
    """Run streaming inference for multiple images"""
    
    # Create client
    triton_client = grpcclient.InferenceServerClient(
        url=server_url,
        verbose=False
    )
    
    print(f"\nStreaming Inference:")
    print(f"Model: {model_name}")
    print(f"Processing {len(image_paths)} images...")
    
    # Create queue for async results
    user_data = queue.Queue()
    
    # Start streaming
    triton_client.start_stream(callback=lambda result, error: callback(user_data, result, error))
    
    # Send multiple requests
    start_time = time.perf_counter()
    
    for idx, image_path in enumerate(image_paths):
        # Preprocess image
        input_data = preprocess_image(image_path)
        
        # Create input object
        inputs = []
        inputs.append(grpcclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
        inputs[0].set_data_from_numpy(input_data)
        
        # Create output object
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('output__0'))
        
        # Send async request
        triton_client.async_stream_infer(
            model_name,
            inputs,
            outputs=outputs,
            request_id=str(idx)
        )
    
    # Collect results
    results = []
    for _ in range(len(image_paths)):
        data_item = user_data.get()
        if not isinstance(data_item, InferenceServerException):
            results.append(data_item)
    
    # Stop streaming
    triton_client.stop_stream()
    
    # Calculate total time
    total_time_ms = (time.perf_counter() - start_time) * 1000
    throughput = len(image_paths) / (total_time_ms / 1000)
    
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Average latency: {total_time_ms/len(image_paths):.2f} ms")
    print(f"Throughput: {throughput:.2f} images/sec")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Triton gRPC Client')
    parser.add_argument('--server', type=str, default='localhost:8001',
                        help='Triton server URL')
    parser.add_argument('--model', type=str, default='resnet50_pytorch',
                        help='Model name')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--async', action='store_true',
                        help='Use async inference')
    parser.add_argument('--stream', action='store_true',
                        help='Use streaming inference')
    
    args = parser.parse_args()
    
    if args.stream:
        # For streaming demo, use same image multiple times
        image_paths = [args.image] * 10
        run_streaming_inference(args.server, args.model, image_paths)
    elif args.async:
        run_async_inference(args.server, args.model, args.image)
    else:
        run_sync_inference(args.server, args.model, args.image)


if __name__ == '__main__':
    main()