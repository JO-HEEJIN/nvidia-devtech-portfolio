#!/usr/bin/env python3
"""
Prepare models for Triton Inference Server deployment
Downloads and converts ResNet50 to TorchScript and TensorRT formats
"""

import os
import sys
import torch
import torchvision
import argparse
from pathlib import Path


def download_and_convert_resnet50_pytorch(output_dir):
    """Download ResNet50 and convert to TorchScript format"""
    print("Downloading ResNet50 model...")
    
    # Download pretrained ResNet50
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(1, 3, 224, 224)
    
    # Convert to TorchScript via tracing
    print("Converting to TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Save model
    output_path = Path(output_dir) / "model_repository" / "resnet50_pytorch" / "1" / "model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    traced_model.save(str(output_path))
    print(f"Saved PyTorch model to {output_path}")
    
    return traced_model


def export_tensorrt_engine(pytorch_model, output_dir):
    """Export TensorRT engine from PyTorch model"""
    print("Exporting TensorRT engine...")
    
    try:
        import torch_tensorrt
        
        # Configure TensorRT conversion
        example_input = torch.randn(1, 3, 224, 224).cuda()
        
        trt_model = torch_tensorrt.compile(
            pytorch_model,
            inputs=[
                torch_tensorrt.Input(
                    shape=[1, 3, 224, 224],
                    dtype=torch.float32
                )
            ],
            enabled_precisions={torch.float32, torch.float16},
            workspace_size=1 << 30,  # 1GB
            truncate_long_and_double=True
        )
        
        # Save TensorRT engine
        output_path = Path(output_dir) / "model_repository" / "resnet50_tensorrt" / "1" / "model.plan"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.jit.save(trt_model, str(output_path))
        print(f"Saved TensorRT engine to {output_path}")
        
    except ImportError:
        print("Warning: torch_tensorrt not installed. Skipping TensorRT export.")
        print("Install with: pip install torch-tensorrt")
        
        # Create dummy file for testing
        output_path = Path(output_dir) / "model_repository" / "resnet50_tensorrt" / "1" / "model.plan"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("DUMMY_TENSORRT_ENGINE")
        print(f"Created dummy TensorRT file at {output_path}")


def validate_model_repository(repo_dir):
    """Validate the model repository structure"""
    print("\nValidating model repository structure...")
    
    required_models = [
        "resnet50_pytorch",
        "resnet50_tensorrt",
        "ensemble_preprocess_infer"
    ]
    
    repo_path = Path(repo_dir) / "model_repository"
    
    for model in required_models:
        model_path = repo_path / model
        config_path = model_path / "config.pbtxt"
        
        if config_path.exists():
            print(f"✓ Found config for {model}")
        else:
            print(f"✗ Missing config for {model}")
            
        # Check for version directories
        version_dirs = list(model_path.glob("[0-9]*/"))
        if version_dirs:
            print(f"  Version directories: {[d.name for d in version_dirs]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare models for Triton deployment")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="..",
        help="Output directory (default: parent directory)"
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT engine export"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    output_dir = Path(args.output_dir).absolute()
    
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Download and convert ResNet50
    pytorch_model = download_and_convert_resnet50_pytorch(output_dir)
    
    # Export TensorRT engine
    if not args.skip_tensorrt:
        if torch.cuda.is_available():
            export_tensorrt_engine(pytorch_model, output_dir)
        else:
            print("CUDA not available. Skipping TensorRT export.")
            # Create dummy file
            output_path = Path(output_dir) / "model_repository" / "resnet50_tensorrt" / "1" / "model.plan"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("DUMMY_TENSORRT_ENGINE")
    
    # Validate repository structure
    validate_model_repository(output_dir)
    
    print("\nModel preparation complete!")


if __name__ == "__main__":
    main()