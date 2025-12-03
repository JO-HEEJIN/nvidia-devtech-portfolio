#!/usr/bin/env python3
"""
Model checkpoint conversion script for TensorRT-LLM optimization.
Downloads HuggingFace models and converts them to TensorRT-LLM checkpoint format.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoConfig,
        LlamaForCausalLM,
        LlamaTokenizer,
        LlamaConfig
    )
    import torch
except ImportError as e:
    logger.error(f"Required dependencies not installed: {e}")
    logger.error("Please run: pip install transformers torch")
    exit(1)

class ModelConverter:
    """Convert HuggingFace models to TensorRT-LLM checkpoint format."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize converter with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        self.quantization_config = config.get('quantization', {})
        self.paths = config.get('paths', {})
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def download_model(self, model_name: str, cache_dir: Optional[str] = None) -> tuple:
        """Download model and tokenizer from HuggingFace."""
        logger.info(f"Downloading model: {model_name}")
        
        try:
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Handle missing pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Download model config
            logger.info("Downloading model config...")
            model_config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Download model weights
            logger.info("Downloading model weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map='auto' if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("Model download completed successfully")
            return model, tokenizer, model_config
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def extract_model_info(self, model, tokenizer, model_config) -> Dict[str, Any]:
        """Extract model architecture information."""
        logger.info("Extracting model information...")
        
        # Get model state dict
        state_dict = model.state_dict()
        
        # Extract architecture info
        info = {
            'model_type': getattr(model_config, 'model_type', 'unknown'),
            'vocab_size': getattr(model_config, 'vocab_size', len(tokenizer)),
            'hidden_size': getattr(model_config, 'hidden_size', 0),
            'num_layers': getattr(model_config, 'num_hidden_layers', 0),
            'num_heads': getattr(model_config, 'num_attention_heads', 0),
            'num_kv_heads': getattr(model_config, 'num_key_value_heads', None),
            'intermediate_size': getattr(model_config, 'intermediate_size', 0),
            'max_position_embeddings': getattr(model_config, 'max_position_embeddings', 2048),
            'rope_base': getattr(model_config, 'rope_theta', 10000.0),
            'norm_epsilon': getattr(model_config, 'rms_norm_eps', 1e-5),
        }
        
        # Handle missing num_kv_heads (for older models)
        if info['num_kv_heads'] is None:
            info['num_kv_heads'] = info['num_heads']
        
        # Calculate model size
        total_params = sum(p.numel() for p in model.parameters())
        info['total_parameters'] = total_params
        info['model_size_mb'] = total_params * 2 / (1024 * 1024)  # FP16
        
        logger.info(f"Model info extracted: {info}")
        return info
    
    def save_checkpoint(self, model, tokenizer, model_config, output_dir: str):
        """Save model in TensorRT-LLM checkpoint format."""
        logger.info(f"Saving checkpoint to: {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = output_path / "pytorch_model.bin"
        logger.info(f"Saving model weights to: {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Save config
        config_path = output_path / "config.json"
        logger.info(f"Saving config to: {config_path}")
        with open(config_path, 'w') as f:
            json.dump(model_config.to_dict(), f, indent=2)
        
        # Save tokenizer
        tokenizer_path = output_path / "tokenizer"
        tokenizer_path.mkdir(exist_ok=True)
        logger.info(f"Saving tokenizer to: {tokenizer_path}")
        tokenizer.save_pretrained(str(tokenizer_path))
        
        # Save model info
        model_info = self.extract_model_info(model, tokenizer, model_config)
        info_path = output_path / "model_info.json"
        logger.info(f"Saving model info to: {info_path}")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Checkpoint saved successfully")
    
    def convert(self, output_dir: Optional[str] = None):
        """Main conversion function."""
        logger.info("Starting model conversion process...")
        
        # Get model name
        model_name = self.model_config.get('hf_model_name')
        if not model_name:
            raise ValueError("hf_model_name not specified in config")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.paths.get('model_dir', './models/converted')
        
        # Create cache directory
        cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # Download model
            model, tokenizer, model_config = self.download_model(
                model_name, 
                cache_dir=cache_dir
            )
            
            # Save checkpoint
            self.save_checkpoint(
                model, 
                tokenizer, 
                model_config, 
                output_dir
            )
            
            logger.info("Model conversion completed successfully!")
            return output_dir
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
    
    def validate_checkpoint(self, checkpoint_dir: str) -> bool:
        """Validate saved checkpoint."""
        logger.info(f"Validating checkpoint: {checkpoint_dir}")
        
        checkpoint_path = Path(checkpoint_dir)
        required_files = [
            'pytorch_model.bin',
            'config.json', 
            'model_info.json',
            'tokenizer/tokenizer.json',
            'tokenizer/tokenizer_config.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not (checkpoint_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing files in checkpoint: {missing_files}")
            return False
        
        # Validate model can be loaded
        try:
            state_dict = torch.load(
                checkpoint_path / 'pytorch_model.bin',
                map_location='cpu'
            )
            logger.info(f"Checkpoint contains {len(state_dict)} parameters")
            
            with open(checkpoint_path / 'config.json') as f:
                config = json.load(f)
            logger.info(f"Model type: {config.get('model_type', 'unknown')}")
            
            logger.info("Checkpoint validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.error("PyYAML not installed. Please run: pip install pyyaml")
        raise
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to TensorRT-LLM checkpoint format"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory for checkpoint (overrides config)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate checkpoint after conversion'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize converter
        converter = ModelConverter(config)
        
        # Convert model
        output_dir = converter.convert(args.output_dir)
        
        # Validate if requested
        if args.validate:
            if converter.validate_checkpoint(output_dir):
                logger.info("Checkpoint validation passed")
            else:
                logger.error("Checkpoint validation failed")
                exit(1)
        
        logger.info(f"Conversion completed successfully! Output: {output_dir}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()