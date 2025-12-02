#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX format for TensorRT conversion
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def export_to_onnx(
    model_name='yolov8s',
    output_path='yolov8s.onnx',
    imgsz=640,
    dynamic_batch=False,
    simplify=True,
    opset=12
):
    """
    Export YOLOv8 model to ONNX format
    
    Args:
        model_name: YOLOv8 model variant (n/s/m/l/x)
        output_path: Output ONNX file path
        imgsz: Input image size
        dynamic_batch: Enable dynamic batch size
        simplify: Simplify ONNX model
        opset: ONNX opset version
    """
    
    # Load YOLOv8 model
    print(f"Loading {model_name} model...")
    if model_name.endswith('.pt'):
        # Load custom trained model
        model = YOLO(model_name)
    else:
        # Load pretrained model
        model = YOLO(f'{model_name}.pt')
    
    # Configure export settings
    print(f"Exporting to ONNX (imgsz={imgsz}, dynamic={dynamic_batch})...")
    
    # Export to ONNX
    success = model.export(
        format='onnx',
        imgsz=imgsz if isinstance(imgsz, list) else [imgsz, imgsz],
        dynamic=dynamic_batch,
        simplify=simplify,
        opset=opset,
        batch=1 if not dynamic_batch else None
    )
    
    if success:
        # Move exported file to desired location
        default_export_path = Path(model_name).with_suffix('.onnx')
        if default_export_path.exists() and default_export_path != Path(output_path):
            default_export_path.rename(output_path)
        
        print(f"Successfully exported to {output_path}")
        
        # Verify ONNX model
        verify_onnx(output_path, imgsz, dynamic_batch)
        
        return True
    else:
        print("Export failed!")
        return False


def verify_onnx(onnx_path, imgsz, dynamic_batch):
    """
    Verify exported ONNX model
    """
    import onnx
    import onnxruntime as ort
    import numpy as np
    
    print("\nVerifying ONNX model...")
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure valid")
    
    # Test inference
    session = ort.InferenceSession(onnx_path)
    
    # Get input details
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    input_name = input_info.name
    
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    
    # Prepare test input
    if dynamic_batch:
        batch_size = 1
    else:
        batch_size = input_shape[0] if input_shape[0] != 'batch' else 1
    
    if isinstance(imgsz, int):
        h = w = imgsz
    else:
        h, w = imgsz
    
    test_input = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: test_input})
    
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Number of detections: {outputs[0].shape[1]}")
    print("  ONNX model verified successfully")
    
    # Print model info
    print("\nModel Information:")
    print(f"  File size: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    print(f"  IR version: {onnx_model.ir_version}")
    
    # Print input/output details
    print("\nInput/Output Details:")
    for i, input_info in enumerate(session.get_inputs()):
        print(f"  Input {i}: {input_info.name} - {input_info.shape} ({input_info.type})")
    
    for i, output_info in enumerate(session.get_outputs()):
        print(f"  Output {i}: {output_info.name} - {output_info.shape} ({output_info.type})")


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 to ONNX')
    parser.add_argument('--model', type=str, default='yolov8s',
                        help='YOLOv8 model variant (n/s/m/l/x) or path to .pt file')
    parser.add_argument('--output', type=str, default='yolov8s.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, nargs='+', default=640,
                        help='Input image size (single value or h w)')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Disable ONNX simplification')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    # Convert image size
    if isinstance(args.imgsz, list):
        if len(args.imgsz) == 1:
            imgsz = args.imgsz[0]
        else:
            imgsz = args.imgsz[:2]
    else:
        imgsz = args.imgsz
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    success = export_to_onnx(
        model_name=args.model,
        output_path=str(output_path),
        imgsz=imgsz,
        dynamic_batch=args.dynamic_batch,
        simplify=not args.no_simplify,
        opset=args.opset
    )
    
    if success:
        print(f"\nExport complete! ONNX model saved to: {output_path}")
        print("\nNext steps:")
        print(f"  1. Build TensorRT engine: python src/build_engine.py --onnx {output_path}")
        print(f"  2. Run inference: python demo/run_image.py --engine <engine_path> --image <image_path>")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()