#!/usr/bin/env python3
"""
Triton HTTP client for inference
Uses tritonclient.http for REST API communication
"""

import numpy as np
import sys
import time
import argparse
from PIL import Image
import tritonclient.http as httpclient
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


def run_inference(server_url, model_name, image_path):
    """Run inference using HTTP client"""
    
    # Create client
    triton_client = httpclient.InferenceServerClient(
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
    print(f"Version: {model_metadata['versions']}")
    print(f"Inputs: {model_metadata['inputs']}")
    print(f"Outputs: {model_metadata['outputs']}")
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Create input object
    inputs = []
    inputs.append(httpclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Create output object
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output__0'))
    
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
    
    print(f"\nInference Results:")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Output shape: {output_data.shape}")
    print(f"Top 5 predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"  {i+1}. Class {idx}: {output_data[0][idx]:.4f}")
    
    return output_data, latency_ms


def main():
    parser = argparse.ArgumentParser(description='Triton HTTP Client')
    parser.add_argument('--server', type=str, default='localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--model', type=str, default='resnet50_pytorch',
                        help='Model name')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(args.server, args.model, args.image)


if __name__ == '__main__':
    main()