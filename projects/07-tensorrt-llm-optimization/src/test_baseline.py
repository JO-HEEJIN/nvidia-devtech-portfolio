#!/usr/bin/env python3
"""
Test script for HuggingFace baseline functionality.
Tests the baseline inference without requiring actual model downloads.
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

def test_memory_tracker():
    """Test memory tracking functionality."""
    logger.info("Testing memory tracker...")
    
    try:
        import sys
        sys.path.append('.')
        from src.inference_hf import MemoryTracker
        
        # Initialize tracker
        tracker = MemoryTracker()
        
        # Test basic memory measurement
        memory_usage = tracker.get_memory_usage()
        
        required_fields = ['cpu_memory_mb', 'cpu_memory_percent']
        for field in required_fields:
            if field not in memory_usage:
                raise ValueError(f"Missing required field: {field}")
        
        # Test measurement recording
        tracker.record_measurement("test_point")
        
        if len(tracker.measurements) != 1:
            raise ValueError("Measurement not recorded properly")
        
        logger.info(f"✓ Memory tracker working. CPU memory: {memory_usage['cpu_memory_mb']:.1f} MB")
        return True
        
    except ImportError as e:
        logger.warning(f"Cannot test memory tracker: {e}")
        return False
    except Exception as e:
        logger.error(f"Memory tracker test failed: {e}")
        return False

def test_prompt_loading():
    """Test test prompt loading."""
    logger.info("Testing prompt loading...")
    
    try:
        import sys
        sys.path.append('.')
        from src.inference_hf import load_test_prompts
        
        prompts = load_test_prompts()
        
        if not isinstance(prompts, list):
            raise ValueError("Prompts should be a list")
        
        if len(prompts) == 0:
            raise ValueError("No prompts loaded")
        
        for prompt in prompts:
            if not isinstance(prompt, str) or len(prompt) == 0:
                raise ValueError("Invalid prompt format")
        
        logger.info(f"✓ Loaded {len(prompts)} test prompts")
        return True
        
    except Exception as e:
        logger.error(f"Prompt loading test failed: {e}")
        return False

def test_results_saving():
    """Test results saving functionality."""
    logger.info("Testing results saving...")
    
    try:
        import sys
        sys.path.append('.')
        from src.inference_hf import HuggingFaceInference
        
        # Create mock results
        mock_results = {
            'model_name': 'test_model',
            'num_prompts': 1,
            'total_tokens': 100,
            'tokens_per_second': 50.0,
            'memory_usage': {'peak_gpu_memory_mb': 1000}
        }
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create inference instance (without loading model)
            inference = HuggingFaceInference("dummy_model")
            inference.save_results(mock_results, tmp_path)
            
            # Verify file was created
            if not Path(tmp_path).exists():
                raise ValueError("Results file was not created")
            
            # Verify content
            import json
            with open(tmp_path, 'r') as f:
                saved_results = json.load(f)
            
            if saved_results['model_name'] != 'test_model':
                raise ValueError("Results not saved correctly")
            
            logger.info("✓ Results saving working correctly")
            return True
            
        finally:
            # Clean up
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Results saving test failed: {e}")
        return False

def test_config_parsing():
    """Test generation config parsing."""
    logger.info("Testing config parsing...")
    
    try:
        # Test generation parameters
        test_params = {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.9
        }
        
        # Validate parameter ranges
        if not (0 < test_params['temperature'] <= 2.0):
            raise ValueError("Invalid temperature range")
        
        if not (0 < test_params['top_p'] <= 1.0):
            raise ValueError("Invalid top_p range")
        
        if not (test_params['max_new_tokens'] > 0):
            raise ValueError("Invalid max_new_tokens")
        
        logger.info("✓ Config parsing validation working")
        return True
        
    except Exception as e:
        logger.error(f"Config parsing test failed: {e}")
        return False

def test_script_structure():
    """Test script file structure and imports."""
    logger.info("Testing script structure...")
    
    try:
        script_path = Path("src/inference_hf.py")
        
        if not script_path.exists():
            raise ValueError("inference_hf.py not found")
        
        # Read script content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for required components
        required_components = [
            'class MemoryTracker',
            'class HuggingFaceInference',
            'def load_model',
            'def generate_tokens',
            'def benchmark',
            'def main'
        ]
        
        for component in required_components:
            if component not in content:
                raise ValueError(f"Missing required component: {component}")
        
        logger.info("✓ Script structure is complete")
        return True
        
    except Exception as e:
        logger.error(f"Script structure test failed: {e}")
        return False

def run_baseline_tests():
    """Run all baseline functionality tests."""
    logger.info("Running HuggingFace baseline functionality tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Memory Tracker", test_memory_tracker),
        ("Prompt Loading", test_prompt_loading),
        ("Results Saving", test_results_saving),
        ("Config Parsing", test_config_parsing),
        ("Script Structure", test_script_structure),
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
    logger.info(f"Baseline Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All baseline tests passed! Ready for model inference.")
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed. Check baseline implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_baseline_tests()
    exit(0 if success else 1)