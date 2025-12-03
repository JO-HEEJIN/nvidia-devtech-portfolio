#!/usr/bin/env python3
"""
Configuration validation script for TensorRT-LLM optimization project.
Validates YAML configuration files and ensures all required parameters are present.
"""

import yaml
import os
import sys
from typing import Dict, List, Any
from pathlib import Path

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Successfully loaded config: {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"✗ YAML parsing error in {config_path}: {e}")
        return {}

def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration section."""
    required_fields = [
        'name', 'hf_model_name', 'architecture', 'vocab_size',
        'hidden_size', 'num_layers', 'num_heads', 'intermediate_size'
    ]
    
    model_config = config.get('model', {})
    missing_fields = [field for field in required_fields if field not in model_config]
    
    if missing_fields:
        print(f"✗ Missing model fields: {missing_fields}")
        return False
    
    print("✓ Model configuration is valid")
    return True

def validate_quantization_config(config: Dict[str, Any]) -> bool:
    """Validate quantization configuration section."""
    required_fields = ['mode', 'precision', 'use_weight_only']
    
    quant_config = config.get('quantization', {})
    missing_fields = [field for field in required_fields if field not in quant_config]
    
    if missing_fields:
        print(f"✗ Missing quantization fields: {missing_fields}")
        return False
    
    # Validate precision values
    valid_modes = ['fp16', 'int8_weight_only', 'int4_weight_only']
    if quant_config.get('mode') not in valid_modes:
        print(f"✗ Invalid quantization mode: {quant_config.get('mode')}")
        return False
    
    print("✓ Quantization configuration is valid")
    return True

def validate_engine_config(config: Dict[str, Any]) -> bool:
    """Validate engine configuration section."""
    required_fields = [
        'max_batch_size', 'max_input_len', 'max_output_len', 'max_seq_len'
    ]
    
    engine_config = config.get('engine', {})
    missing_fields = [field for field in required_fields if field not in engine_config]
    
    if missing_fields:
        print(f"✗ Missing engine fields: {missing_fields}")
        return False
    
    # Validate numerical constraints
    max_seq_len = engine_config.get('max_seq_len', 0)
    max_input_len = engine_config.get('max_input_len', 0)
    max_output_len = engine_config.get('max_output_len', 0)
    
    if max_input_len + max_output_len > max_seq_len:
        print(f"✗ max_input_len + max_output_len ({max_input_len + max_output_len}) > max_seq_len ({max_seq_len})")
        return False
    
    print("✓ Engine configuration is valid")
    return True

def validate_paths_config(config: Dict[str, Any]) -> bool:
    """Validate paths configuration section."""
    required_fields = ['model_dir', 'engine_dir', 'tokenizer_dir', 'output_dir']
    
    paths_config = config.get('paths', {})
    missing_fields = [field for field in required_fields if field not in paths_config]
    
    if missing_fields:
        print(f"✗ Missing paths fields: {missing_fields}")
        return False
    
    print("✓ Paths configuration is valid")
    return True

def validate_single_config(config_path: str) -> bool:
    """Validate a single configuration file."""
    print(f"\nValidating {config_path}...")
    
    config = load_yaml_config(config_path)
    if not config:
        return False
    
    validations = [
        validate_model_config(config),
        validate_quantization_config(config),
        validate_engine_config(config),
        validate_paths_config(config)
    ]
    
    is_valid = all(validations)
    
    if is_valid:
        print(f"✓ {config_path} is fully valid")
    else:
        print(f"✗ {config_path} has validation errors")
    
    return is_valid

def main():
    """Main validation function."""
    print("TensorRT-LLM Configuration Validator")
    print("=" * 40)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    configs_dir = project_root / "configs"
    
    # List of configuration files to validate
    config_files = [
        "tinyllama_fp16.yaml",
        "tinyllama_int8.yaml", 
        "tinyllama_int4.yaml"
    ]
    
    all_valid = True
    
    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            is_valid = validate_single_config(str(config_path))
            all_valid = all_valid and is_valid
        else:
            print(f"✗ Configuration file not found: {config_path}")
            all_valid = False
    
    print("\n" + "=" * 40)
    if all_valid:
        print("✓ All configuration files are valid!")
        sys.exit(0)
    else:
        print("✗ Some configuration files have errors!")
        sys.exit(1)

if __name__ == "__main__":
    main()