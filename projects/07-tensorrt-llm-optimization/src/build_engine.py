#!/usr/bin/env python3
"""
TensorRT engine builder for TensorRT-LLM optimization.
Builds optimized TensorRT engines from converted checkpoints.
"""

import os
import json
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import tensorrt as trt
except ImportError as e:
    logger.warning(f"Some dependencies not available: {e}")
    logger.warning("This may be normal if TensorRT-LLM is not fully installed")

class EngineBuilder:
    """Build TensorRT-LLM engines from model checkpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize engine builder with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        self.quantization_config = config.get('quantization', {})
        self.engine_config = config.get('engine', {})
        self.optimization_config = config.get('optimization', {})
        self.paths = config.get('paths', {})
        
        # Check CUDA availability
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available CUDA devices: {self.device_count}")
    
    def prepare_build_command(self, checkpoint_dir: str, output_dir: str) -> List[str]:
        """Prepare TensorRT-LLM build command."""
        logger.info("Preparing build command...")
        
        # Base command - using trtllm-build (new format)
        cmd = ["trtllm-build"]
        
        # Model architecture
        model_type = self.model_config.get('architecture', 'llama')
        cmd.extend(["--model_type", model_type])
        
        # Input/output paths
        cmd.extend(["--checkpoint_dir", checkpoint_dir])
        cmd.extend(["--output_dir", output_dir])
        
        # Model parameters
        cmd.extend(["--max_batch_size", str(self.engine_config.get('max_batch_size', 8))])
        cmd.extend(["--max_input_len", str(self.engine_config.get('max_input_len', 1024))])
        cmd.extend(["--max_output_len", str(self.engine_config.get('max_output_len', 512))])
        cmd.extend(["--max_beam_width", str(self.engine_config.get('max_beam_width', 1))])
        
        # Quantization settings
        quant_mode = self.quantization_config.get('mode', 'fp16')
        if quant_mode == 'fp16':
            cmd.append("--dtype=float16")
        elif quant_mode == 'int8_weight_only':
            cmd.append("--use_weight_only")
            cmd.append("--weight_only_precision=int8")
        elif quant_mode == 'int4_weight_only':
            cmd.append("--use_weight_only")
            cmd.append("--weight_only_precision=int4")
            
            # AWQ settings
            if self.quantization_config.get('use_awq', False):
                cmd.append("--use_awq")
                group_size = self.quantization_config.get('group_size', 128)
                cmd.extend(["--group_size", str(group_size)])
        
        # KV Cache settings
        if self.engine_config.get('use_paged_kv_cache', True):
            cmd.append("--paged_kv_cache")
            tokens_per_block = self.engine_config.get('tokens_per_block', 64)
            cmd.extend(["--tokens_per_block", str(tokens_per_block)])
        
        # Attention plugins
        if self.engine_config.get('use_gpt_attention_plugin', True):
            cmd.append("--gpt_attention_plugin=float16")
        
        if self.engine_config.get('use_gemm_plugin', True):
            cmd.append("--gemm_plugin=float16")
        
        # Memory optimizations
        if self.optimization_config.get('remove_input_padding', True):
            cmd.append("--remove_input_padding")
        
        if self.optimization_config.get('use_packed_input', True):
            cmd.append("--use_packed_input")
        
        # Multi-GPU settings (if available)
        if self.device_count > 1:
            cmd.extend(["--world_size", str(self.device_count)])
            cmd.extend(["--tp_size", str(self.device_count)])
        
        # Performance settings
        if self.optimization_config.get('enable_xqa', True):
            cmd.append("--use_custom_all_reduce=disable")
        
        if self.engine_config.get('use_inflight_batching', False):
            cmd.append("--use_inflight_batching")
        
        # Verbose output
        cmd.append("--log_level=verbose")
        
        logger.info(f"Build command: {' '.join(cmd)}")
        return cmd
    
    def build_engine(self, checkpoint_dir: str, output_dir: str) -> bool:
        """Build TensorRT engine."""
        logger.info(f"Building engine from {checkpoint_dir} to {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare build command
            cmd = self.prepare_build_command(checkpoint_dir, output_dir)
            
            # Execute build command
            logger.info("Starting engine build process...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Log output
            if result.stdout:
                logger.info(f"Build stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Build stderr:\n{result.stderr}")
            
            if result.returncode == 0:
                logger.info("Engine build completed successfully")
                return True
            else:
                logger.error(f"Engine build failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Engine build timed out")
            return False
        except FileNotFoundError:
            logger.error("trtllm-build command not found. Is TensorRT-LLM properly installed?")
            return False
        except Exception as e:
            logger.error(f"Error during engine build: {e}")
            return False
    
    def save_engine_metadata(self, engine_dir: str, checkpoint_dir: str):
        """Save engine metadata for later use."""
        logger.info("Saving engine metadata...")
        
        metadata = {
            'model_config': self.model_config,
            'quantization_config': self.quantization_config,
            'engine_config': self.engine_config,
            'optimization_config': self.optimization_config,
            'source_checkpoint': checkpoint_dir,
            'build_timestamp': str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else 'CPU'
        }
        
        metadata_path = Path(engine_dir) / 'engine_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def validate_engine(self, engine_dir: str) -> bool:
        """Validate built engine."""
        logger.info(f"Validating engine: {engine_dir}")
        
        engine_path = Path(engine_dir)
        
        # Check for required files
        required_files = [
            'config.json',
            'model.engine'  # May vary based on TensorRT-LLM version
        ]
        
        # Alternative engine file patterns
        engine_files = list(engine_path.glob('*.engine'))
        plan_files = list(engine_path.glob('*.plan'))
        
        if not engine_files and not plan_files:
            logger.error("No engine or plan files found")
            return False
        
        # Check config file
        config_path = engine_path / 'config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                logger.info(f"Engine config loaded: {config.get('model_type', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not read engine config: {e}")
        
        logger.info("Engine validation passed")
        return True
    
    def build(self, checkpoint_dir: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        """Main build function."""
        logger.info("Starting engine build process...")
        
        # Set default paths
        if checkpoint_dir is None:
            checkpoint_dir = self.paths.get('model_dir', './models/converted')
        if output_dir is None:
            output_dir = self.paths.get('engine_dir', './engines/built')
        
        # Validate checkpoint directory
        if not Path(checkpoint_dir).exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        try:
            # Build engine
            success = self.build_engine(checkpoint_dir, output_dir)
            if not success:
                raise RuntimeError("Engine build failed")
            
            # Save metadata
            self.save_engine_metadata(output_dir, checkpoint_dir)
            
            # Validate engine
            if not self.validate_engine(output_dir):
                logger.warning("Engine validation failed, but build completed")
            
            logger.info(f"Engine build completed successfully! Output: {output_dir}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Build process failed: {e}")
            raise

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
        description="Build TensorRT-LLM engines from checkpoints"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--checkpoint_dir',
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory for engine'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate engine after build'
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
        
        # Initialize builder
        builder = EngineBuilder(config)
        
        # Build engine
        output_dir = builder.build(args.checkpoint_dir, args.output_dir)
        
        logger.info(f"Build completed successfully! Engine: {output_dir}")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()