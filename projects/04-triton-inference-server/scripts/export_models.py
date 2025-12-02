#!/usr/bin/env python3
"""
Export and prepare models for Triton Inference Server
"""

import argparse
import os
import sys
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Optional


def export_resnet50_pytorch(output_path: str):
    """
    Export ResNet50 PyTorch model
    
    Args:
        output_path: Path to save model.pt
    """
    print("Exporting ResNet50 (PyTorch)...")
    
    try:
        import torchvision.models as models
        
        # Load pretrained ResNet50
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        traced_model.save(output_path)
        
        print(f"  ✓ Saved ResNet50 to {output_path}")
        print(f"  Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to export ResNet50: {e}")
        return False


def export_yolov8_tensorrt(output_path: str, precision: str = "FP16"):
    """
    Export YOLOv8 to TensorRT engine
    
    Args:
        output_path: Path to save model.plan
        precision: TensorRT precision (FP32, FP16, INT8)
    """
    print(f"Exporting YOLOv8 (TensorRT {precision})...")
    
    try:
        # Check if TensorRT is available
        import tensorrt as trt
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')  # Using nano version for demo
        
        # Export to TensorRT
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export with TensorRT optimization
        model.export(
            format='engine',
            device='cuda',
            half=(precision == 'FP16'),
            int8=(precision == 'INT8'),
            workspace=4,  # GB
            batch=16,
            verbose=False
        )
        
        # Move the exported engine
        engine_path = 'yolov8n.engine'
        if os.path.exists(engine_path):
            shutil.move(engine_path, output_path)
            print(f"  ✓ Saved YOLOv8 TensorRT engine to {output_path}")
            print(f"  Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            raise FileNotFoundError("TensorRT engine was not created")
            
    except ImportError as e:
        print(f"  ✗ Dependencies missing: {e}")
        print("  Install with: pip install ultralytics tensorrt")
        return False
    except Exception as e:
        print(f"  ✗ Failed to export YOLOv8: {e}")
        print("  Falling back to ONNX export...")
        return export_yolov8_onnx(output_path.replace('.plan', '.onnx'))


def export_yolov8_onnx(output_path: str):
    """
    Export YOLOv8 to ONNX (fallback if TensorRT unavailable)
    
    Args:
        output_path: Path to save model.onnx
    """
    print("Exporting YOLOv8 (ONNX)...")
    
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Export to ONNX
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model.export(
            format='onnx',
            imgsz=640,
            batch=16,
            opset=12,
            verbose=False
        )
        
        # Move the exported model
        onnx_path = 'yolov8n.onnx'
        if os.path.exists(onnx_path):
            shutil.move(onnx_path, output_path)
            print(f"  ✓ Saved YOLOv8 ONNX to {output_path}")
            print(f"  Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            raise FileNotFoundError("ONNX model was not created")
            
    except ImportError as e:
        print(f"  ✗ Dependencies missing: {e}")
        print("  Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"  ✗ Failed to export YOLOv8: {e}")
        return False


def export_bert_onnx(output_path: str, max_length: int = 128):
    """
    Export BERT to ONNX for text classification
    
    Args:
        output_path: Path to save model.onnx
        max_length: Maximum sequence length
    """
    print(f"Exporting BERT (ONNX, max_length={max_length})...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Load pretrained BERT model
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        model.eval()
        
        # Create dummy inputs
        dummy_text = "This is a test sentence."
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        
        # Export to ONNX
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=12,
            do_constant_folding=True
        )
        
        print(f"  ✓ Saved BERT ONNX to {output_path}")
        print(f"  Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        # Save tokenizer config
        tokenizer_path = os.path.join(os.path.dirname(output_path), "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"  ✓ Saved tokenizer to {tokenizer_path}")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Dependencies missing: {e}")
        print("  Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"  ✗ Failed to export BERT: {e}")
        return False


def create_dummy_model(output_path: str, model_type: str):
    """
    Create dummy model file for testing
    
    Args:
        output_path: Path to save dummy model
        model_type: Type of model (pt, plan, onnx)
    """
    print(f"Creating dummy {model_type} model...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if model_type == "pt":
        # Create dummy PyTorch model
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3 * 224 * 224, 1000)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1))
        
        model = DummyModel()
        traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        traced.save(output_path)
        
    elif model_type == "onnx":
        # Create dummy ONNX model
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model
        input_tensor = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, [1, 3, 224, 224]
        )
        output_tensor = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, [1, 1000]
        )
        
        # Create a dummy weight
        weight = helper.make_tensor(
            'weight',
            TensorProto.FLOAT,
            [1000, 3 * 224 * 224],
            vals=np.random.randn(1000, 3 * 224 * 224).flatten().tolist()
        )
        
        # Create MatMul node
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input_flat', 'weight_t'],
            outputs=['output']
        )
        
        # Create graph
        graph = helper.make_graph(
            [matmul_node],
            'dummy_model',
            [input_tensor],
            [output_tensor],
            [weight]
        )
        
        # Create model
        model = helper.make_model(graph)
        onnx.save(model, output_path)
        
    else:  # plan (TensorRT)
        # Create dummy binary file
        with open(output_path, 'wb') as f:
            f.write(b'DUMMY_TENSORRT_ENGINE_' + os.urandom(1024))
    
    print(f"  ✓ Created dummy {model_type} model at {output_path}")


def verify_model_repository(repo_path: str):
    """
    Verify model repository structure
    
    Args:
        repo_path: Path to model repository
    """
    print("\nVerifying model repository structure...")
    
    required_models = [
        "resnet50_pytorch",
        "yolov8_tensorrt",
        "bert_onnx"
    ]
    
    all_valid = True
    
    for model in required_models:
        model_path = os.path.join(repo_path, model)
        config_path = os.path.join(model_path, "config.pbtxt")
        version_path = os.path.join(model_path, "1")
        
        # Check structure
        if not os.path.exists(config_path):
            print(f"  ✗ Missing config for {model}: {config_path}")
            all_valid = False
        else:
            print(f"  ✓ Found config for {model}")
        
        if not os.path.exists(version_path):
            print(f"  ✗ Missing version directory for {model}: {version_path}")
            all_valid = False
        else:
            # Check for model file
            model_files = os.listdir(version_path)
            if model_files:
                print(f"  ✓ Found model files for {model}: {model_files}")
            else:
                print(f"  ✗ No model files in {version_path}")
                all_valid = False
    
    return all_valid


def optimize_models(repo_path: str):
    """
    Optimize models for deployment
    
    Args:
        repo_path: Path to model repository
    """
    print("\nOptimizing models for deployment...")
    
    # ResNet50 optimization
    resnet_path = os.path.join(repo_path, "resnet50_pytorch", "1", "model.pt")
    if os.path.exists(resnet_path):
        size_mb = os.path.getsize(resnet_path) / 1024 / 1024
        print(f"  ResNet50: {size_mb:.2f} MB")
        if size_mb > 100:
            print("    Consider using quantization to reduce size")
    
    # YOLOv8 optimization
    yolo_path = os.path.join(repo_path, "yolov8_tensorrt", "1", "model.plan")
    if os.path.exists(yolo_path):
        size_mb = os.path.getsize(yolo_path) / 1024 / 1024
        print(f"  YOLOv8: {size_mb:.2f} MB")
        if size_mb > 50:
            print("    Consider using FP16 or INT8 precision")
    
    # BERT optimization
    bert_path = os.path.join(repo_path, "bert_onnx", "1", "model.onnx")
    if os.path.exists(bert_path):
        size_mb = os.path.getsize(bert_path) / 1024 / 1024
        print(f"  BERT: {size_mb:.2f} MB")
        if size_mb > 500:
            print("    Consider using DistilBERT or quantization")


def main():
    parser = argparse.ArgumentParser(description='Export models for Triton')
    parser.add_argument('--repo', type=str, default='../model_repository',
                        help='Model repository path')
    parser.add_argument('--model', type=str, choices=['resnet50', 'yolov8', 'bert', 'all'],
                        default='all', help='Model to export')
    parser.add_argument('--dummy', action='store_true',
                        help='Create dummy models for testing')
    parser.add_argument('--precision', type=str, choices=['FP32', 'FP16', 'INT8'],
                        default='FP16', help='TensorRT precision')
    parser.add_argument('--verify', action='store_true',
                        help='Verify repository structure')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize models')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    repo_path = os.path.abspath(args.repo)
    
    print(f"Model Repository: {repo_path}")
    print("-" * 50)
    
    success_count = 0
    total_count = 0
    
    # Export models
    if args.model in ['resnet50', 'all']:
        total_count += 1
        model_path = os.path.join(repo_path, "resnet50_pytorch", "1", "model.pt")
        
        if args.dummy:
            create_dummy_model(model_path, "pt")
            success_count += 1
        else:
            if export_resnet50_pytorch(model_path):
                success_count += 1
    
    if args.model in ['yolov8', 'all']:
        total_count += 1
        model_path = os.path.join(repo_path, "yolov8_tensorrt", "1", "model.plan")
        
        if args.dummy:
            create_dummy_model(model_path, "plan")
            success_count += 1
        else:
            if export_yolov8_tensorrt(model_path, args.precision):
                success_count += 1
    
    if args.model in ['bert', 'all']:
        total_count += 1
        model_path = os.path.join(repo_path, "bert_onnx", "1", "model.onnx")
        
        if args.dummy:
            create_dummy_model(model_path, "onnx")
            success_count += 1
        else:
            if export_bert_onnx(model_path):
                success_count += 1
    
    print("-" * 50)
    print(f"Export Summary: {success_count}/{total_count} models exported successfully")
    
    # Verify structure
    if args.verify:
        if verify_model_repository(repo_path):
            print("\n✓ Model repository is valid")
        else:
            print("\n✗ Model repository has issues")
    
    # Optimize models
    if args.optimize:
        optimize_models(repo_path)
    
    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())