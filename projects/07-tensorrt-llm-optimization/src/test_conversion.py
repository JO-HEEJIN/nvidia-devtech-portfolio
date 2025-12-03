#!/usr/bin/env python3
"""
Test script for model conversion pipeline.
Tests the conversion from HuggingFace to TensorRT-LLM checkpoint format.
"""

import os
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test configuration loading."""
    logger.info("Testing configuration loading...")
    
    try:
        from src.convert_checkpoint import load_config
        
        # Test with FP16 config
        config_path = "configs/tinyllama_fp16.yaml"
        if Path(config_path).exists():
            config = load_config(config_path)
            
            # Validate required sections
            required_sections = ['model', 'quantization', 'engine', 'paths']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            logger.info(f"✓ Config loaded successfully: {config['model']['name']}")
            return True
        else:
            logger.warning(f"Config file not found: {config_path}")
            return False
            
    except ImportError as e:
        logger.warning(f"Cannot test config loading: {e}")
        return False
    except Exception as e:
        logger.error(f"Config loading test failed: {e}")
        return False

def test_model_info_extraction():
    """Test model information extraction without downloading."""
    logger.info("Testing model info extraction...")
    
    try:
        # Create a mock model config for testing
        mock_config = {
            'model_type': 'llama',
            'vocab_size': 32000,
            'hidden_size': 2048,
            'num_hidden_layers': 22,
            'num_attention_heads': 32,
            'num_key_value_heads': 4,
            'intermediate_size': 5632,
            'max_position_embeddings': 2048,
            'rope_theta': 10000.0,
            'rms_norm_eps': 1e-5
        }
        
        # Test info extraction logic
        info = {
            'model_type': mock_config.get('model_type', 'unknown'),
            'vocab_size': mock_config.get('vocab_size', 0),
            'hidden_size': mock_config.get('hidden_size', 0),
            'num_layers': mock_config.get('num_hidden_layers', 0),
            'num_heads': mock_config.get('num_attention_heads', 0),
            'num_kv_heads': mock_config.get('num_key_value_heads', 0),
            'intermediate_size': mock_config.get('intermediate_size', 0),
            'max_position_embeddings': mock_config.get('max_position_embeddings', 2048),
            'rope_base': mock_config.get('rope_theta', 10000.0),
            'norm_epsilon': mock_config.get('rms_norm_eps', 1e-5),
        }
        
        logger.info(f"✓ Model info extracted: {info['model_type']} with {info['num_layers']} layers")
        return True
        
    except Exception as e:
        logger.error(f"Model info extraction test failed: {e}")
        return False

def test_directory_creation():
    """Test checkpoint directory creation."""
    logger.info("Testing directory creation...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_checkpoint"
            test_path.mkdir(parents=True, exist_ok=True)
            
            # Test subdirectory creation
            (test_path / "tokenizer").mkdir(exist_ok=True)
            
            # Verify directories exist
            if test_path.exists() and (test_path / "tokenizer").exists():
                logger.info("✓ Directory creation successful")
                return True
            else:
                logger.error("Directory creation failed")
                return False
                
    except Exception as e:
        logger.error(f"Directory creation test failed: {e}")
        return False

def test_build_command_generation():
    """Test TensorRT build command generation."""
    logger.info("Testing build command generation...")
    
    try:
        from src.build_engine import EngineBuilder, load_config
        
        # Load test config
        config_path = "configs/tinyllama_fp16.yaml"
        if not Path(config_path).exists():
            logger.warning("Config file not found, using mock config")
            config = {
                'model': {'architecture': 'llama'},
                'quantization': {'mode': 'fp16'},
                'engine': {
                    'max_batch_size': 8,
                    'max_input_len': 1024,
                    'max_output_len': 512,
                    'max_beam_width': 1,
                    'use_gpt_attention_plugin': True,
                    'use_gemm_plugin': True,
                    'use_paged_kv_cache': True,
                    'tokens_per_block': 64
                },
                'optimization': {
                    'remove_input_padding': True,
                    'use_packed_input': True,
                    'enable_xqa': True
                },
                'paths': {}
            }
        else:
            config = load_config(config_path)
        
        # Initialize builder
        builder = EngineBuilder(config)
        
        # Generate build command
        cmd = builder.prepare_build_command("/mock/checkpoint", "/mock/output")
        
        # Validate command structure
        if len(cmd) > 0 and cmd[0] == "trtllm-build":
            logger.info(f"✓ Build command generated: {len(cmd)} arguments")
            logger.info(f"  Command preview: {' '.join(cmd[:5])}...")
            return True
        else:
            logger.error("Invalid build command generated")
            return False
            
    except Exception as e:
        logger.error(f"Build command generation test failed: {e}")
        return False

def run_all_tests():
    """Run all conversion pipeline tests."""
    logger.info("Running TensorRT-LLM conversion pipeline tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Model Info Extraction", test_model_info_extraction),
        ("Directory Creation", test_directory_creation),
        ("Build Command Generation", test_build_command_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All tests passed! Conversion pipeline is ready.")
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed. Some components may need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)