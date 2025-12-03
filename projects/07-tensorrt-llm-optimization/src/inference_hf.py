#!/usr/bin/env python3
"""
HuggingFace baseline inference for TensorRT-LLM optimization comparison.
Provides baseline performance metrics for tokens per second and memory usage.
"""

import os
import time
import json
import argparse
import logging
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoConfig,
        GenerationConfig
    )
    from transformers.utils import logging as transformers_logging
    
    # Set transformers logging to warning to reduce noise
    transformers_logging.set_verbosity_warning()
    
except ImportError as e:
    logger.error(f"Required dependencies not installed: {e}")
    logger.error("Please run: pip install torch transformers")
    exit(1)

class MemoryTracker:
    """Track memory usage during inference."""
    
    def __init__(self):
        """Initialize memory tracker."""
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        self.measurements = []
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = {
            'cpu_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_memory_percent': self.process.memory_percent(),
        }
        
        if self.gpu_available:
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            })
        
        return memory_info
    
    def record_measurement(self, label: str):
        """Record a memory measurement with label."""
        memory_usage = self.get_memory_usage()
        memory_usage['label'] = label
        memory_usage['timestamp'] = time.time()
        self.measurements.append(memory_usage)
        logger.debug(f"Memory [{label}]: {memory_usage}")
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage across all measurements."""
        if not self.measurements:
            return {}
        
        peak_usage = {}
        for key in self.measurements[0].keys():
            if key in ['label', 'timestamp']:
                continue
            peak_usage[f'peak_{key}'] = max(m[key] for m in self.measurements)
        
        return peak_usage
    
    def reset_gpu_stats(self):
        """Reset GPU memory statistics."""
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

class HuggingFaceInference:
    """HuggingFace baseline inference implementation."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize HuggingFace inference."""
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.generation_config = None
        self.memory_tracker = MemoryTracker()
        
        # Performance metrics
        self.metrics = {
            'total_tokens_generated': 0,
            'total_time_seconds': 0.0,
            'tokens_per_second': 0.0,
            'time_to_first_token': 0.0,
            'inter_token_latency': 0.0
        }
    
    def load_model(self, torch_dtype: str = "float16", use_cache: bool = True):
        """Load HuggingFace model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.memory_tracker.reset_gpu_stats()
        self.memory_tracker.record_measurement("before_model_load")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Handle missing pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load config
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Determine torch dtype
            if torch_dtype == "float16":
                dtype = torch.float16
            elif torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Load model
            logger.info("Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
                use_cache=use_cache,
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            self.memory_tracker.record_measurement("after_model_load")
            
            # Setup generation config
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=use_cache
            )
            
            logger.info("Model loaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Model dtype: {dtype}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_tokens(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        measure_timing: bool = True
    ) -> Dict[str, Any]:
        """Generate tokens with detailed timing measurements."""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating {max_new_tokens} tokens for prompt: {prompt[:50]}...")
        
        # Reset memory tracking
        self.memory_tracker.record_measurement("before_generation")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        logger.info(f"Input length: {input_length} tokens")
        
        # Generation timing
        timing_data = []
        generated_tokens = []
        
        # Configure generation
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": False
        }
        
        try:
            # Start generation
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    **generation_kwargs
                )
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            self.memory_tracker.record_measurement("after_generation")
            
            # Decode generated text
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            actual_new_tokens = len(generated_ids)
            
            # Calculate metrics
            tokens_per_second = actual_new_tokens / total_time if total_time > 0 else 0
            
            # Update overall metrics
            self.metrics['total_tokens_generated'] += actual_new_tokens
            self.metrics['total_time_seconds'] += total_time
            self.metrics['tokens_per_second'] = (
                self.metrics['total_tokens_generated'] / self.metrics['total_time_seconds']
                if self.metrics['total_time_seconds'] > 0 else 0
            )
            
            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'full_text': prompt + generated_text,
                'input_tokens': input_length,
                'output_tokens': actual_new_tokens,
                'total_time_seconds': total_time,
                'tokens_per_second': tokens_per_second,
                'generation_config': {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p
                }
            }
            
            logger.info(f"Generated {actual_new_tokens} tokens in {total_time:.3f}s")
            logger.info(f"Tokens per second: {tokens_per_second:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def benchmark(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Run benchmark on multiple prompts."""
        logger.info(f"Running benchmark on {len(prompts)} prompts")
        
        results = []
        total_tokens = 0
        total_time = 0.0
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            try:
                result = self.generate_tokens(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                results.append(result)
                total_tokens += result['output_tokens']
                total_time += result['total_time_seconds']
                
            except Exception as e:
                logger.error(f"Error processing prompt {i+1}: {e}")
                continue
        
        # Calculate overall metrics
        avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        # Get memory statistics
        peak_memory = self.memory_tracker.get_peak_usage()
        
        benchmark_results = {
            'model_name': self.model_name,
            'num_prompts': len(prompts),
            'total_input_tokens': sum(r['input_tokens'] for r in results),
            'total_output_tokens': total_tokens,
            'total_time_seconds': total_time,
            'average_tokens_per_second': avg_tokens_per_second,
            'individual_results': results,
            'memory_usage': peak_memory,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'model_type': self.config.model_type if self.config else 'unknown',
                'torch_dtype': str(next(self.model.parameters()).dtype),
                'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
            }
        }
        
        logger.info(f"Benchmark completed: {avg_tokens_per_second:.2f} tokens/second")
        return benchmark_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")

def load_test_prompts() -> List[str]:
    """Load test prompts for benchmarking."""
    return [
        "The future of artificial intelligence is",
        "In a world where technology has advanced beyond our wildest dreams,",
        "Explain the concept of machine learning in simple terms:",
        "Write a short story about a robot learning to feel emotions:",
        "What are the key differences between supervised and unsupervised learning?",
        "Describe the potential impact of quantum computing on society:",
        "Create a dialogue between two AI systems discussing consciousness:",
        "Explain how neural networks work using everyday analogies:",
    ]

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="HuggingFace baseline inference for TensorRT-LLM comparison"
    )
    parser.add_argument(
        '--model_name',
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--output',
        default='results/hf_baseline_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    parser.add_argument(
        '--torch_dtype',
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Torch dtype for model'
    )
    parser.add_argument(
        '--custom_prompt',
        help='Custom prompt to test'
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
        # Initialize inference
        inference = HuggingFaceInference(args.model_name, args.device)
        
        # Load model
        inference.load_model(torch_dtype=args.torch_dtype)
        
        # Prepare prompts
        if args.custom_prompt:
            prompts = [args.custom_prompt]
        else:
            prompts = load_test_prompts()
        
        # Run benchmark
        results = inference.benchmark(
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Save results
        inference.save_results(results, args.output)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("HUGGINGFACE BASELINE RESULTS")
        logger.info("="*50)
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Total tokens generated: {results['total_output_tokens']}")
        logger.info(f"Average tokens per second: {results['average_tokens_per_second']:.2f}")
        logger.info(f"Peak GPU memory: {results['memory_usage'].get('peak_gpu_memory_allocated_mb', 'N/A')} MB")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()