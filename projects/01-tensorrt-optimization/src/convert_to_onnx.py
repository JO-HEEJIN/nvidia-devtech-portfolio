#!/usr/bin/env python3
"""
PyTorch to ONNX Model Converter

This module handles the conversion of PyTorch models to ONNX format,
preparing them for TensorRT optimization. The ONNX intermediate representation
allows for model portability and serves as the input format for TensorRT.

Key features:
- Dynamic batch size support for flexible inference
- Model validation after export
- Support for custom and pretrained models
- Comprehensive error handling and logging
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import onnx
import onnxruntime
import numpy as np
from coloredlogs import install as setup_colored_logs


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging with colored output for better visibility.
    
    Args:
        verbose: Enable debug level logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    setup_colored_logs(
        level=level,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_pytorch_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a PyTorch model from torchvision or a custom path.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50') or path to saved model
        pretrained: Whether to load pretrained weights for torchvision models
        
    Returns:
        Loaded PyTorch model in evaluation mode
        
    Raises:
        ValueError: If model cannot be loaded
    """
    logger = logging.getLogger(__name__)
    
    # Check if model_name is a file path
    if os.path.exists(model_name):
        logger.info(f"Loading custom model from: {model_name}")
        model = torch.load(model_name, map_location='cpu')
        if isinstance(model, dict):
            # Assume it's a state dict, need to know the architecture
            raise ValueError("State dict provided without model architecture")
    else:
        # Load from torchvision
        logger.info(f"Loading torchvision model: {model_name}")
        if not hasattr(models, model_name):
            available = [m for m in dir(models) if not m.startswith('_')]
            raise ValueError(f"Model {model_name} not found. Available: {available[:10]}...")
        
        model_func = getattr(models, model_name)
        model = model_func(pretrained=pretrained)
    
    model.eval()
    logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    dynamic_batch: bool = False,
    opset_version: int = 16,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None
) -> None:
    """
    Export PyTorch model to ONNX format with configurable options.
    
    This function handles the core conversion process, managing:
    - Dynamic axis specification for variable batch sizes
    - Operator set version compatibility
    - Model optimization during export
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Shape of model input (batch_size, channels, height, width)
        dynamic_batch: Enable dynamic batch size support
        opset_version: ONNX opset version for operator compatibility
        input_names: Names for input tensors
        output_names: Names for output tensors
        
    Raises:
        RuntimeError: If export fails
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input for tracing
    dummy_input = torch.randn(*input_shape)
    
    # Set input/output names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Configure dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        # The batch dimension (0) can vary at runtime
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
        logger.info("Dynamic batch size enabled")
    
    logger.info(f"Exporting model to ONNX format...")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Opset version: {opset_version}")
    
    try:
        # Export model
        # do_constant_folding optimizes the graph by evaluating constant expressions
        # export_params includes the trained parameters in the exported model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logger.info(f"ONNX export successful: {output_path}")
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"ONNX model size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        raise RuntimeError(f"Failed to export model to ONNX: {str(e)}")


def validate_onnx_model(onnx_path: str, input_shape: Tuple[int, ...]) -> bool:
    """
    Validate exported ONNX model for correctness and compatibility.
    
    Performs two levels of validation:
    1. ONNX checker - Verifies model structure and operators
    2. Runtime validation - Ensures model can be executed
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Expected input shape for validation
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load and check model structure
        logger.info("Validating ONNX model structure...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model structure is valid")
        
        # Validate with ONNX Runtime
        logger.info("Testing ONNX model inference...")
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # Get input details
        input_info = ort_session.get_inputs()[0]
        logger.info(f"Model input: name={input_info.name}, shape={input_info.shape}, type={input_info.type}")
        
        # Get output details  
        output_info = ort_session.get_outputs()[0]
        logger.info(f"Model output: name={output_info.name}, shape={output_info.shape}, type={output_info.type}")
        
        # Run test inference
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = ort_session.run(None, {input_info.name: dummy_input})
        
        logger.info(f"Test inference successful. Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {str(e)}")
        return False


def compare_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """
    Compare outputs between PyTorch and ONNX models to ensure conversion accuracy.
    
    This verification step is crucial for ensuring the conversion process
    maintains model accuracy. Small numerical differences are expected due
    to different computation backends.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        input_shape: Input tensor shape
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        
    Returns:
        True if outputs match within tolerance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Generate test input
        test_input = torch.randn(*input_shape)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # ONNX Runtime inference
        ort_session = onnxruntime.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: test_input.numpy()})[0]
        
        # Compare outputs
        matches = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        
        if matches:
            logger.info("Output validation PASSED: PyTorch and ONNX outputs match")
            
            # Calculate actual differences
            diff = np.abs(pytorch_output - onnx_output)
            logger.info(f"Max absolute difference: {diff.max():.2e}")
            logger.info(f"Mean absolute difference: {diff.mean():.2e}")
        else:
            logger.warning("Output validation FAILED: Outputs differ beyond tolerance")
            diff = np.abs(pytorch_output - onnx_output)
            logger.warning(f"Max absolute difference: {diff.max():.2e}")
            logger.warning(f"Mean absolute difference: {diff.mean():.2e}")
        
        return matches
        
    except Exception as e:
        logger.error(f"Output comparison failed: {str(e)}")
        return False


def main():
    """Main entry point for ONNX conversion."""
    
    parser = argparse.ArgumentParser(
        description='Convert PyTorch models to ONNX format for TensorRT optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        help='Model name from torchvision or path to saved model'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models/resnet50.onnx',
        help='Output path for ONNX model'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for export (use 1 for dynamic batching)'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Input image size (height width)'
    )
    
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='Number of input channels'
    )
    
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Enable dynamic batch size support'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=16,
        help='ONNX opset version'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after export'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("PyTorch to ONNX Converter")
    logger.info("="*60)
    
    try:
        # Load model
        model = load_pytorch_model(args.model)
        
        # Prepare input shape
        input_shape = (
            args.batch_size,
            args.channels,
            args.input_size[0],
            args.input_size[1]
        )
        
        # Export to ONNX
        export_to_onnx(
            model=model,
            output_path=args.output,
            input_shape=input_shape,
            dynamic_batch=args.dynamic_batch,
            opset_version=args.opset
        )
        
        # Validate if requested
        if not args.no_validate:
            logger.info("-"*60)
            logger.info("Running validation...")
            
            # Structural validation
            if not validate_onnx_model(args.output, input_shape):
                logger.error("Validation failed")
                sys.exit(1)
            
            # Output comparison
            if not compare_outputs(model, args.output, input_shape):
                logger.warning("Output comparison showed differences")
        
        logger.info("="*60)
        logger.info("Conversion completed successfully!")
        logger.info(f"ONNX model saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()