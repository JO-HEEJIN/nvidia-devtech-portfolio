#!/usr/bin/env python3
"""
TensorRT-LLM inference implementation for optimized LLM inference.
Provides TensorRT-LLM based inference with streaming, batching, and KV cache management.
"""

import os
import time
import json
import argparse
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer
except ImportError as e:
    logger.error(f"Basic dependencies not installed: {e}")
    logger.error("Please run: pip install torch transformers")
    exit(1)

# TensorRT-LLM imports (may not be available during initial setup)
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelConfig, SamplingConfig
    from tensorrt_llm import Mapping
    from tensorrt_llm.runtime import Session
    from tensorrt_llm.bindings import GptSession
    TRTLLM_AVAILABLE = True
except ImportError:
    logger.warning("TensorRT-LLM not available. Some functionality will be limited.")
    TRTLLM_AVAILABLE = False

class TensorRTLLMInference:
    """TensorRT-LLM optimized inference implementation."""
    
    def __init__(self, engine_dir: str, tokenizer_path: Optional[str] = None):
        """Initialize TensorRT-LLM inference."""
        self.engine_dir = Path(engine_dir)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.session = None
        self.model_config = None
        self.sampling_config = None
        
        # Performance tracking
        self.metrics = {
            'total_tokens_generated': 0,
            'total_time_seconds': 0.0,
            'tokens_per_second': 0.0,
            'time_to_first_token': 0.0,
            'inter_token_latency': 0.0
        }
    
    def load_engine_and_tokenizer(self):
        """Load TensorRT-LLM engine and tokenizer."""
        logger.info(f"Loading TensorRT-LLM engine from: {self.engine_dir}")
        
        if not TRTLLM_AVAILABLE:
            raise RuntimeError("TensorRT-LLM not available. Please install tensorrt-llm package.")
        
        try:
            # Load engine metadata
            metadata_path = self.engine_dir / 'engine_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Engine metadata loaded: {metadata.get('model_config', {}).get('name', 'unknown')}")
            
            # Load model configuration
            config_path = self.engine_dir / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    engine_config = json.load(f)
                logger.info("Engine config loaded")
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Initialize model config
            self.model_config = ModelConfig(
                vocab_size=engine_config.get('vocab_size', 32000),
                num_layers=engine_config.get('num_layers', 22),
                num_heads=engine_config.get('num_heads', 32),
                hidden_size=engine_config.get('hidden_size', 2048),
                gpt_attention_plugin=engine_config.get('gpt_attention_plugin', True),
                remove_input_padding=engine_config.get('remove_input_padding', True),
                model_name=engine_config.get('model_type', 'llama'),
                use_gpt_attention_plugin=engine_config.get('use_gpt_attention_plugin', True),
                paged_kv_cache=engine_config.get('paged_kv_cache', True),
                tokens_per_block=engine_config.get('tokens_per_block', 64),
                use_custom_all_reduce=engine_config.get('use_custom_all_reduce', False)
            )
            
            # Load tokenizer
            if self.tokenizer_path:
                tokenizer_dir = self.tokenizer_path
            else:
                tokenizer_dir = self.engine_dir / 'tokenizer'
                if not tokenizer_dir.exists():
                    # Fallback to default TinyLlama tokenizer
                    tokenizer_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            logger.info(f"Loading tokenizer from: {tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize TensorRT-LLM session
            self.session = Session.from_serialized_engine(str(self.engine_dir))
            
            logger.info("TensorRT-LLM engine and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading engine: {e}")
            raise
    
    def create_sampling_config(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 100
    ) -> 'SamplingConfig':
        """Create sampling configuration."""
        if not TRTLLM_AVAILABLE:
            raise RuntimeError("TensorRT-LLM not available")
        
        sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            num_beams=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=1.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens
        )
        
        return sampling_config
    
    def generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Generate tokens using TensorRT-LLM."""
        
        if self.session is None:
            raise ValueError("Engine not loaded. Call load_engine_and_tokenizer() first.")
        
        logger.info(f"Generating {max_new_tokens} tokens for prompt: {prompt[:50]}...")
        
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            input_length = input_ids.shape[1]
            
            logger.info(f"Input length: {input_length} tokens")
            
            # Create sampling config
            sampling_config = self.create_sampling_config(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens
            )
            
            # Start generation timing
            start_time = time.perf_counter()
            first_token_time = None
            
            if streaming:
                # Streaming generation
                output_ids = []
                for step, token_id in enumerate(self._generate_streaming(input_ids, sampling_config)):
                    output_ids.append(token_id)
                    
                    # Record first token time
                    if step == 0 and first_token_time is None:
                        first_token_time = time.perf_counter() - start_time
                    
                    # Stop if EOS token or max length reached
                    if token_id == self.tokenizer.eos_token_id or len(output_ids) >= max_new_tokens:
                        break
                
                output_ids = torch.tensor([output_ids])
            else:
                # Non-streaming generation
                output_ids = self.session.generate(
                    input_ids,
                    sampling_config=sampling_config,
                    streaming=False
                )
                first_token_time = time.perf_counter() - start_time
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Decode generated text
            if output_ids.dim() > 1:
                generated_ids = output_ids[0]
            else:
                generated_ids = output_ids
            
            # Remove input tokens from output
            if len(generated_ids) > input_length:
                new_token_ids = generated_ids[input_length:]
            else:
                new_token_ids = generated_ids
            
            generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            actual_new_tokens = len(new_token_ids)
            
            # Calculate metrics
            tokens_per_second = actual_new_tokens / total_time if total_time > 0 else 0
            
            if first_token_time is not None:
                inter_token_latency = (total_time - first_token_time) / max(1, actual_new_tokens - 1)
            else:
                inter_token_latency = 0
            
            # Update metrics
            self.metrics['total_tokens_generated'] += actual_new_tokens
            self.metrics['total_time_seconds'] += total_time
            self.metrics['tokens_per_second'] = (
                self.metrics['total_tokens_generated'] / self.metrics['total_time_seconds']
                if self.metrics['total_time_seconds'] > 0 else 0
            )
            self.metrics['time_to_first_token'] = first_token_time or 0
            self.metrics['inter_token_latency'] = inter_token_latency
            
            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'full_text': prompt + generated_text,
                'input_tokens': input_length,
                'output_tokens': actual_new_tokens,
                'total_time_seconds': total_time,
                'tokens_per_second': tokens_per_second,
                'time_to_first_token': first_token_time or 0,
                'inter_token_latency': inter_token_latency,
                'generation_config': {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k,
                    'streaming': streaming
                }
            }
            
            logger.info(f"Generated {actual_new_tokens} tokens in {total_time:.3f}s")
            logger.info(f"Tokens per second: {tokens_per_second:.2f}")
            logger.info(f"Time to first token: {first_token_time or 0:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _generate_streaming(self, input_ids: torch.Tensor, sampling_config) -> torch.Tensor:
        """Generator for streaming token generation."""
        # This is a placeholder for actual TensorRT-LLM streaming implementation
        # The actual implementation would depend on the specific TensorRT-LLM API
        
        logger.warning("Streaming generation is a placeholder implementation")
        
        # Fallback to non-streaming for now
        output_ids = self.session.generate(
            input_ids,
            sampling_config=sampling_config,
            streaming=False
        )
        
        # Simulate streaming by yielding tokens one by one
        if output_ids.dim() > 1:
            output_ids = output_ids[0]
        
        input_length = input_ids.shape[1]
        for i in range(input_length, len(output_ids)):
            yield output_ids[i].item()
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate tokens for batch of prompts."""
        logger.info(f"Processing batch of {len(prompts)} prompts")
        
        results = []
        
        try:
            # For now, process sequentially
            # A full implementation would use TensorRT-LLM batch processing
            for i, prompt in enumerate(prompts):
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                
                result = self.generate_tokens(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise
        
        logger.info(f"Batch generation completed for {len(results)} prompts")
        return results
    
    def benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        use_batch: bool = False
    ) -> Dict[str, Any]:
        """Run benchmark on multiple prompts."""
        logger.info(f"Running TensorRT-LLM benchmark on {len(prompts)} prompts")
        
        start_time = time.perf_counter()
        
        if use_batch:
            results = self.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        else:
            results = []
            for prompt in prompts:
                result = self.generate_tokens(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                results.append(result)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics
        total_input_tokens = sum(r['input_tokens'] for r in results)
        total_output_tokens = sum(r['output_tokens'] for r in results)
        avg_tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
        avg_ttft = sum(r['time_to_first_token'] for r in results) / len(results)
        avg_itl = sum(r['inter_token_latency'] for r in results) / len(results)
        
        benchmark_results = {
            'engine_type': 'tensorrt_llm',
            'engine_dir': str(self.engine_dir),
            'num_prompts': len(prompts),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_time_seconds': total_time,
            'average_tokens_per_second': avg_tokens_per_second,
            'average_time_to_first_token': avg_ttft,
            'average_inter_token_latency': avg_itl,
            'use_batch': use_batch,
            'individual_results': results,
            'model_info': {
                'model_config': self.model_config.__dict__ if self.model_config else {},
                'engine_path': str(self.engine_dir)
            }
        }
        
        logger.info(f"TensorRT-LLM benchmark completed:")
        logger.info(f"  Average TPS: {avg_tokens_per_second:.2f}")
        logger.info(f"  Average TTFT: {avg_ttft:.3f}s")
        logger.info(f"  Average ITL: {avg_itl:.3f}s")
        
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
        description="TensorRT-LLM inference for optimized LLM performance"
    )
    parser.add_argument(
        '--engine_dir',
        required=True,
        help='Path to TensorRT-LLM engine directory'
    )
    parser.add_argument(
        '--tokenizer_path',
        help='Path to tokenizer (default: use engine directory)'
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
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--output',
        default='results/trtllm_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--custom_prompt',
        help='Custom prompt to test'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable streaming generation'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Use batch processing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not TRTLLM_AVAILABLE:
        logger.error("TensorRT-LLM not available. Please install tensorrt-llm package.")
        logger.error("You can still run the HuggingFace baseline with inference_hf.py")
        exit(1)
    
    try:
        # Initialize TensorRT-LLM inference
        inference = TensorRTLLMInference(args.engine_dir, args.tokenizer_path)
        
        # Load engine and tokenizer
        inference.load_engine_and_tokenizer()
        
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
            top_p=args.top_p,
            top_k=args.top_k,
            use_batch=args.batch
        )
        
        # Save results
        inference.save_results(results, args.output)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("TENSORRT-LLM RESULTS")
        logger.info("="*50)
        logger.info(f"Engine: {results['engine_dir']}")
        logger.info(f"Total tokens generated: {results['total_output_tokens']}")
        logger.info(f"Average tokens per second: {results['average_tokens_per_second']:.2f}")
        logger.info(f"Average TTFT: {results['average_time_to_first_token']:.3f}s")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"TensorRT-LLM inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()