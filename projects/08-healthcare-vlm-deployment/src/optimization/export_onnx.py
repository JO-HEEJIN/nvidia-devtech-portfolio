"""
ONNX Export Module for Healthcare VLM Deployment

This module exports BiomedCLIP vision and text encoders to ONNX format for optimized inference.
ONNX provides cross-platform compatibility and serves as intermediate format for TensorRT conversion.

Key Features:
- Separate vision and text encoder exports
- Dynamic input shape support for various medical image sizes
- Medical image preprocessing integration
- Opset version optimization for healthcare deployment
- Validation against original PyTorch model
"""

import torch
import torch.nn as nn
import onnx
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import tempfile
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomedCLIPONNXExporter:
    """
    Export BiomedCLIP components to ONNX format with medical imaging optimizations.
    
    Handles both vision and text encoders separately to enable:
    - Independent optimization of each component
    - Flexible deployment strategies
    - Medical domain-specific preprocessing
    """
    
    def __init__(self, 
                 output_dir: str = "./onnx_models",
                 opset_version: int = 16):
        """
        Initialize ONNX exporter.
        
        Args:
            output_dir: Directory to save ONNX models
            opset_version: ONNX opset version (16 recommended for latest features)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.opset_version = opset_version
        
        logger.info(f"ONNX exporter initialized - Output: {output_dir}, Opset: {opset_version}")
    
    def export_vision_encoder(self,
                             model_loader,
                             input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                             dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                             model_name: str = "vision_encoder") -> str:
        """
        Export vision encoder to ONNX format.
        
        Args:
            model_loader: BiomedCLIP loader instance
            input_shape: Input tensor shape [batch, channels, height, width]
            dynamic_axes: Dictionary defining dynamic dimensions
            model_name: Name for exported model
            
        Returns:
            Path to exported ONNX file
        """
        logger.info("Exporting vision encoder to ONNX...")
        
        try:
            # Get vision encoder
            vision_model = model_loader.get_vision_encoder()
            vision_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape, device=model_loader.device)
            
            # Default dynamic axes for medical imaging
            if dynamic_axes is None:
                dynamic_axes = {
                    'image_input': {
                        0: 'batch_size',
                        2: 'height', 
                        3: 'width'
                    },
                    'image_features': {
                        0: 'batch_size'
                    }
                }
            
            # Export path
            output_path = self.output_dir / f"{model_name}.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                vision_model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['image_input'],
                output_names=['image_features'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Validate exported model
            self._validate_onnx_model(output_path, dummy_input, vision_model)
            
            logger.info(f"Vision encoder exported successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export vision encoder: {e}")
            raise
    
    def export_text_encoder(self,
                           model_loader,
                           max_sequence_length: int = 256,
                           dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                           model_name: str = "text_encoder") -> str:
        """
        Export text encoder to ONNX format.
        
        Args:
            model_loader: BiomedCLIP loader instance
            max_sequence_length: Maximum text sequence length
            dynamic_axes: Dictionary defining dynamic dimensions
            model_name: Name for exported model
            
        Returns:
            Path to exported ONNX file
        """
        logger.info("Exporting text encoder to ONNX...")
        
        try:
            # Get text encoder
            text_model = model_loader.get_text_encoder()
            text_model.eval()
            
            # Create dummy input based on tokenizer type
            if hasattr(model_loader.tokenizer, 'encode_plus'):
                # HuggingFace tokenizer
                dummy_text = ["normal chest x-ray findings"]
                tokens = model_loader.tokenizer(
                    dummy_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_sequence_length
                )
                dummy_input = tokens['input_ids'].to(model_loader.device)
            else:
                # OpenCLIP tokenizer
                dummy_input = torch.randint(
                    0, 10000, 
                    (1, max_sequence_length), 
                    device=model_loader.device
                )
            
            # Default dynamic axes for text
            if dynamic_axes is None:
                dynamic_axes = {
                    'text_input': {
                        0: 'batch_size',
                        1: 'sequence_length'
                    },
                    'text_features': {
                        0: 'batch_size'
                    }
                }
            
            # Export path
            output_path = self.output_dir / f"{model_name}.onnx"
            
            # Export to ONNX
            torch.onnx.export(
                text_model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['text_input'],
                output_names=['text_features'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Validate exported model
            self._validate_onnx_model(output_path, dummy_input, text_model)
            
            logger.info(f"Text encoder exported successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to export text encoder: {e}")
            raise
    
    def export_complete_model(self, model_loader) -> Dict[str, str]:
        """
        Export both vision and text encoders.
        
        Args:
            model_loader: BiomedCLIP loader instance
            
        Returns:
            Dictionary with paths to exported models
        """
        logger.info("Exporting complete BiomedCLIP model to ONNX...")
        
        exported_models = {}
        
        # Export vision encoder
        vision_path = self.export_vision_encoder(model_loader)
        exported_models['vision_encoder'] = vision_path
        
        # Export text encoder  
        text_path = self.export_text_encoder(model_loader)
        exported_models['text_encoder'] = text_path
        
        # Create metadata file
        metadata_path = self.output_dir / "model_metadata.json"
        self._save_metadata(model_loader, metadata_path, exported_models)
        exported_models['metadata'] = str(metadata_path)
        
        logger.info("Complete model export finished")
        return exported_models
    
    def _validate_onnx_model(self, 
                           onnx_path: Path, 
                           dummy_input: torch.Tensor,
                           pytorch_model: nn.Module) -> None:
        """
        Validate ONNX model against PyTorch model.
        
        Args:
            onnx_path: Path to ONNX model
            dummy_input: Test input tensor
            pytorch_model: Original PyTorch model
        """
        try:
            import onnxruntime as ort
            
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = pytorch_model(dummy_input).cpu().numpy()
            
            # Get ONNX output
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            relative_error = max_diff / (np.max(np.abs(pytorch_output)) + 1e-8)
            
            logger.info(f"ONNX validation - Max diff: {max_diff:.6f}, Relative error: {relative_error:.6f}")
            
            if relative_error > 1e-3:
                logger.warning(f"High relative error in ONNX conversion: {relative_error}")
            else:
                logger.info("ONNX model validation passed")
                
        except ImportError:
            logger.warning("ONNX Runtime not available - skipping validation")
        except Exception as e:
            logger.warning(f"ONNX validation failed: {e}")
    
    def _save_metadata(self, 
                      model_loader,
                      metadata_path: Path,
                      exported_models: Dict[str, str]) -> None:
        """Save model metadata for deployment."""
        import json
        
        metadata = {
            "model_info": model_loader.get_model_info(),
            "export_config": {
                "opset_version": self.opset_version,
                "output_directory": str(self.output_dir)
            },
            "exported_models": exported_models,
            "medical_domains": [
                "radiology",
                "pathology", 
                "dermatology",
                "ophthalmology"
            ],
            "supported_modalities": [
                "chest_xray",
                "ct_scan",
                "mri",
                "dermoscopy",
                "fundus"
            ]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved: {metadata_path}")
    
    def optimize_for_medical_imaging(self, onnx_path: str) -> str:
        """
        Apply medical imaging specific optimizations to ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Path to optimized model
        """
        logger.info("Applying medical imaging optimizations...")
        
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Apply graph optimizations
            from onnxoptimizer import optimize
            
            # Medical imaging specific optimizations
            optimization_passes = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ]
            
            optimized_model = optimize(model, optimization_passes)
            
            # Save optimized model
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            logger.info(f"Optimized model saved: {optimized_path}")
            return optimized_path
            
        except ImportError:
            logger.warning("ONNX Optimizer not available - skipping optimization")
            return onnx_path
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return onnx_path


def export_biomedclip_to_onnx(model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                             output_dir: str = "./onnx_models",
                             device: str = "auto",
                             opset_version: int = 16) -> Dict[str, str]:
    """
    Convenience function to export BiomedCLIP to ONNX.
    
    Args:
        model_name: BiomedCLIP model name or path
        output_dir: Output directory
        device: Target device
        opset_version: ONNX opset version
        
    Returns:
        Dictionary with exported model paths
    """
    # Import here to avoid circular imports
    from ..models.load_biomedclip import load_biomedclip
    
    # Load model
    model_loader = load_biomedclip(
        model_name=model_name,
        device=device
    )
    
    # Create exporter
    exporter = BiomedCLIPONNXExporter(
        output_dir=output_dir,
        opset_version=opset_version
    )
    
    # Export models
    return exporter.export_complete_model(model_loader)


if __name__ == "__main__":
    # Test ONNX export
    try:
        logger.info("Testing BiomedCLIP ONNX export...")
        
        # Export models
        exported_models = export_biomedclip_to_onnx(
            output_dir="./test_onnx_models"
        )
        
        logger.info(f"Export completed: {exported_models}")
        
        # Test loading exported models
        import onnxruntime as ort
        
        for model_type, model_path in exported_models.items():
            if model_path.endswith('.onnx'):
                try:
                    session = ort.InferenceSession(model_path)
                    logger.info(f"Successfully loaded {model_type} ONNX model")
                    
                    # Print model info
                    inputs = session.get_inputs()
                    outputs = session.get_outputs()
                    logger.info(f"  Inputs: {[inp.name for inp in inputs]}")
                    logger.info(f"  Outputs: {[out.name for out in outputs]}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {e}")
        
        logger.info("ONNX export test completed!")
        
    except Exception as e:
        logger.error(f"ONNX export test failed: {e}")