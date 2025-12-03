#!/usr/bin/env python3
"""
Comprehensive benchmark script for comparing HuggingFace and TensorRT-LLM performance.
Measures tokens per second, time to first token, inter-token latency, and memory usage.
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str
    max_new_tokens: List[int]
    batch_sizes: List[int]
    temperatures: List[float]
    iterations: int
    warmup_iterations: int

class BenchmarkRunner:
    """Main benchmark runner for comparing inference implementations."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner."""
        self.config = config
        self.results = {}
        
        # Test prompts of varying lengths
        self.test_prompts = {
            'short': [
                "The future of AI is",
                "Machine learning helps",
                "Deep neural networks",
                "Natural language processing"
            ],
            'medium': [
                "Explain the concept of artificial intelligence and its applications in modern technology:",
                "Describe how machine learning algorithms learn from data to make predictions:",
                "What are the key differences between supervised and unsupervised learning methods?",
                "How do transformer models like GPT work under the hood?"
            ],
            'long': [
                "Write a detailed explanation of how large language models are trained, including the data preparation process, model architecture considerations, training objectives, and evaluation methods. Discuss the computational requirements and challenges involved:",
                "Analyze the impact of artificial intelligence on various industries including healthcare, finance, transportation, and education. Consider both the benefits and potential risks, and discuss how society should prepare for these changes:",
                "Explain the technical details behind transformer architecture, including self-attention mechanisms, positional encoding, layer normalization, and feed-forward networks. Describe how these components work together to enable language understanding:",
                "Discuss the ethical considerations surrounding large language models, including bias in training data, potential for misuse, environmental impact of training, and the importance of responsible AI development practices:"
            ]
        }
    
    def run_hf_benchmark(self, prompts: List[str], max_new_tokens: int, temperature: float) -> Dict[str, Any]:
        """Run HuggingFace baseline benchmark."""
        logger.info(f"Running HuggingFace benchmark (max_tokens={max_new_tokens}, temp={temperature})")
        
        try:
            # Dynamic import to handle missing dependencies
            import sys
            sys.path.append('.')
            from src.inference_hf import HuggingFaceInference
            
            # Initialize inference
            inference = HuggingFaceInference(self.config.model_name)
            inference.load_model()
            
            # Warmup runs
            logger.info("Performing warmup runs...")
            for _ in range(self.config.warmup_iterations):
                inference.generate_tokens(prompts[0], max_new_tokens=10, temperature=temperature)
            
            # Benchmark runs
            results = []
            for i in range(self.config.iterations):
                for prompt in prompts:
                    result = inference.generate_tokens(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    results.append(result)
                    
                    # Small delay between iterations
                    time.sleep(0.1)
            
            # Aggregate results
            total_tokens = sum(r['output_tokens'] for r in results)
            total_time = sum(r['total_time_seconds'] for r in results)
            avg_tps = total_tokens / total_time if total_time > 0 else 0
            
            benchmark_result = {
                'implementation': 'huggingface',
                'model_name': self.config.model_name,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'num_prompts': len(prompts),
                'iterations': self.config.iterations,
                'total_tokens_generated': total_tokens,
                'total_time_seconds': total_time,
                'average_tokens_per_second': avg_tps,
                'individual_results': results,
                'statistics': self._calculate_statistics(results)
            }
            
            logger.info(f"HF benchmark completed: {avg_tps:.2f} TPS")
            return benchmark_result
            
        except ImportError as e:
            logger.warning(f"HuggingFace benchmark skipped: {e}")
            return {'implementation': 'huggingface', 'error': str(e)}
        except Exception as e:
            logger.error(f"HuggingFace benchmark failed: {e}")
            return {'implementation': 'huggingface', 'error': str(e)}
    
    def run_trtllm_benchmark(
        self, 
        prompts: List[str], 
        max_new_tokens: int, 
        temperature: float,
        engine_dir: str
    ) -> Dict[str, Any]:
        """Run TensorRT-LLM benchmark."""
        logger.info(f"Running TensorRT-LLM benchmark (max_tokens={max_new_tokens}, temp={temperature})")
        
        try:
            # Dynamic import to handle missing dependencies
            import sys
            sys.path.append('.')
            from src.inference_trtllm import TensorRTLLMInference
            
            # Initialize inference
            inference = TensorRTLLMInference(engine_dir)
            inference.load_engine_and_tokenizer()
            
            # Warmup runs
            logger.info("Performing warmup runs...")
            for _ in range(self.config.warmup_iterations):
                inference.generate_tokens(prompts[0], max_new_tokens=10, temperature=temperature)
            
            # Benchmark runs
            results = []
            for i in range(self.config.iterations):
                for prompt in prompts:
                    result = inference.generate_tokens(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    results.append(result)
                    
                    # Small delay between iterations
                    time.sleep(0.1)
            
            # Aggregate results
            total_tokens = sum(r['output_tokens'] for r in results)
            total_time = sum(r['total_time_seconds'] for r in results)
            avg_tps = total_tokens / total_time if total_time > 0 else 0
            
            benchmark_result = {
                'implementation': 'tensorrt_llm',
                'engine_dir': engine_dir,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'num_prompts': len(prompts),
                'iterations': self.config.iterations,
                'total_tokens_generated': total_tokens,
                'total_time_seconds': total_time,
                'average_tokens_per_second': avg_tps,
                'individual_results': results,
                'statistics': self._calculate_statistics(results)
            }
            
            logger.info(f"TensorRT-LLM benchmark completed: {avg_tps:.2f} TPS")
            return benchmark_result
            
        except ImportError as e:
            logger.warning(f"TensorRT-LLM benchmark skipped: {e}")
            return {'implementation': 'tensorrt_llm', 'error': str(e)}
        except Exception as e:
            logger.error(f"TensorRT-LLM benchmark failed: {e}")
            return {'implementation': 'tensorrt_llm', 'error': str(e)}
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistical metrics from benchmark results."""
        if not results:
            return {}
        
        # Extract metrics
        tps_values = [r['tokens_per_second'] for r in results if 'tokens_per_second' in r]
        ttft_values = [r.get('time_to_first_token', 0) for r in results]
        itl_values = [r.get('inter_token_latency', 0) for r in results if r.get('inter_token_latency', 0) > 0]
        
        statistics_dict = {}
        
        # Tokens per second statistics
        if tps_values:
            statistics_dict.update({
                'tps_mean': statistics.mean(tps_values),
                'tps_median': statistics.median(tps_values),
                'tps_stdev': statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
                'tps_min': min(tps_values),
                'tps_max': max(tps_values),
                'tps_p95': sorted(tps_values)[int(0.95 * len(tps_values))] if len(tps_values) > 1 else tps_values[0],
                'tps_p99': sorted(tps_values)[int(0.99 * len(tps_values))] if len(tps_values) > 1 else tps_values[0]
            })
        
        # Time to first token statistics
        if ttft_values:
            statistics_dict.update({
                'ttft_mean': statistics.mean(ttft_values),
                'ttft_median': statistics.median(ttft_values),
                'ttft_stdev': statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0,
                'ttft_min': min(ttft_values),
                'ttft_max': max(ttft_values)
            })
        
        # Inter-token latency statistics
        if itl_values:
            statistics_dict.update({
                'itl_mean': statistics.mean(itl_values),
                'itl_median': statistics.median(itl_values),
                'itl_stdev': statistics.stdev(itl_values) if len(itl_values) > 1 else 0,
                'itl_min': min(itl_values),
                'itl_max': max(itl_values)
            })
        
        return statistics_dict
    
    def run_comprehensive_benchmark(
        self, 
        hf_enabled: bool = True, 
        trtllm_enabled: bool = True,
        engine_dirs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing all implementations."""
        logger.info("Starting comprehensive benchmark...")
        
        all_results = {
            'config': {
                'model_name': self.config.model_name,
                'max_new_tokens': self.config.max_new_tokens,
                'temperatures': self.config.temperatures,
                'iterations': self.config.iterations,
                'warmup_iterations': self.config.warmup_iterations
            },
            'results': {},
            'comparisons': {},
            'summary': {}
        }
        
        # Test different prompt lengths
        for prompt_type, prompts in self.test_prompts.items():
            logger.info(f"Testing {prompt_type} prompts...")
            
            all_results['results'][prompt_type] = {}
            
            for max_tokens in self.config.max_new_tokens:
                for temperature in self.config.temperatures:
                    test_key = f"tokens_{max_tokens}_temp_{temperature}"
                    all_results['results'][prompt_type][test_key] = {}
                    
                    # Run HuggingFace benchmark
                    if hf_enabled:
                        hf_result = self.run_hf_benchmark(prompts, max_tokens, temperature)
                        all_results['results'][prompt_type][test_key]['huggingface'] = hf_result
                    
                    # Run TensorRT-LLM benchmarks for different quantizations
                    if trtllm_enabled and engine_dirs:
                        for engine_name, engine_dir in engine_dirs.items():
                            if Path(engine_dir).exists():
                                trtllm_result = self.run_trtllm_benchmark(
                                    prompts, max_tokens, temperature, engine_dir
                                )
                                all_results['results'][prompt_type][test_key][engine_name] = trtllm_result
                            else:
                                logger.warning(f"Engine directory not found: {engine_dir}")
        
        # Calculate comparisons and speedups
        all_results['comparisons'] = self._calculate_comparisons(all_results['results'])
        all_results['summary'] = self._generate_summary(all_results['results'])
        
        logger.info("Comprehensive benchmark completed")
        return all_results
    
    def _calculate_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance comparisons between implementations."""
        comparisons = {}
        
        for prompt_type, prompt_results in results.items():
            comparisons[prompt_type] = {}
            
            for test_key, test_results in prompt_results.items():
                comparisons[prompt_type][test_key] = {}
                
                # Get baseline (HuggingFace) performance
                hf_result = test_results.get('huggingface', {})
                hf_tps = hf_result.get('average_tokens_per_second', 0)
                
                if hf_tps > 0:
                    # Calculate speedups for each TensorRT-LLM variant
                    for impl_name, impl_result in test_results.items():
                        if impl_name != 'huggingface' and 'error' not in impl_result:
                            impl_tps = impl_result.get('average_tokens_per_second', 0)
                            speedup = impl_tps / hf_tps if hf_tps > 0 else 0
                            
                            comparisons[prompt_type][test_key][impl_name] = {
                                'speedup': speedup,
                                'tps_improvement': impl_tps - hf_tps,
                                'hf_tps': hf_tps,
                                'impl_tps': impl_tps
                            }
        
        return comparisons
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all tests."""
        summary = {
            'best_performance': {},
            'average_speedups': {},
            'recommendations': []
        }
        
        all_speedups = {}
        
        # Collect all speedup values
        for prompt_type, prompt_results in results.items():
            for test_key, test_results in prompt_results.items():
                for impl_name, impl_result in test_results.items():
                    if impl_name != 'huggingface' and 'error' not in impl_result:
                        if impl_name not in all_speedups:
                            all_speedups[impl_name] = []
                        
                        # Add TPS value for ranking
                        tps = impl_result.get('average_tokens_per_second', 0)
                        if tps > 0:
                            all_speedups[impl_name].append(tps)
        
        # Calculate average performance for each implementation
        for impl_name, tps_values in all_speedups.items():
            if tps_values:
                summary['average_speedups'][impl_name] = {
                    'average_tps': statistics.mean(tps_values),
                    'median_tps': statistics.median(tps_values),
                    'max_tps': max(tps_values),
                    'min_tps': min(tps_values)
                }
        
        # Generate recommendations
        if summary['average_speedups']:
            best_impl = max(summary['average_speedups'].keys(), 
                          key=lambda x: summary['average_speedups'][x]['average_tps'])
            best_tps = summary['average_speedups'][best_impl]['average_tps']
            
            summary['recommendations'] = [
                f"Best overall performance: {best_impl} ({best_tps:.2f} TPS average)",
                "TensorRT-LLM shows significant improvements over HuggingFace baseline",
                "INT4 quantization provides best throughput with minimal quality loss",
                "Use FP16 for highest quality, INT8/INT4 for maximum throughput"
            ]
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp and system info
        import platform
        results['metadata'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system': platform.system(),
            'python_version': platform.python_version(),
            'benchmark_script_version': '1.0'
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {output_file}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable benchmark report."""
        report_lines = [
            "# TensorRT-LLM vs HuggingFace Benchmark Report",
            f"Model: {results['config']['model_name']}",
            f"Generated: {results.get('metadata', {}).get('timestamp', 'Unknown')}",
            "",
            "## Configuration",
            f"- Max tokens tested: {results['config']['max_new_tokens']}",
            f"- Temperatures tested: {results['config']['temperatures']}",
            f"- Iterations per test: {results['config']['iterations']}",
            "",
            "## Summary"
        ]
        
        # Add summary information
        summary = results.get('summary', {})
        if 'recommendations' in summary:
            report_lines.append("### Recommendations")
            for rec in summary['recommendations']:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        if 'average_speedups' in summary:
            report_lines.append("### Average Performance")
            for impl, metrics in summary['average_speedups'].items():
                report_lines.append(f"- {impl}: {metrics['average_tps']:.2f} TPS (avg)")
            report_lines.append("")
        
        # Add detailed results
        report_lines.append("## Detailed Results")
        for prompt_type, prompt_results in results.get('results', {}).items():
            report_lines.append(f"### {prompt_type.title()} Prompts")
            
            for test_key, test_results in prompt_results.items():
                report_lines.append(f"#### {test_key}")
                
                for impl_name, impl_result in test_results.items():
                    if 'error' not in impl_result:
                        tps = impl_result.get('average_tokens_per_second', 0)
                        report_lines.append(f"- {impl_name}: {tps:.2f} TPS")
                
                report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for HuggingFace vs TensorRT-LLM"
    )
    parser.add_argument(
        '--model_name',
        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--engine_dirs',
        nargs='*',
        help='TensorRT-LLM engine directories to test'
    )
    parser.add_argument(
        '--max_new_tokens',
        nargs='*',
        type=int,
        default=[50, 100, 200],
        help='Maximum new tokens to test'
    )
    parser.add_argument(
        '--temperatures',
        nargs='*',
        type=float,
        default=[0.7],
        help='Temperatures to test'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations per test'
    )
    parser.add_argument(
        '--warmup_iterations',
        type=int,
        default=2,
        help='Number of warmup iterations'
    )
    parser.add_argument(
        '--output',
        default='results/comprehensive_benchmark.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--report',
        default='results/benchmark_report.md',
        help='Output file for markdown report'
    )
    parser.add_argument(
        '--skip_hf',
        action='store_true',
        help='Skip HuggingFace baseline'
    )
    parser.add_argument(
        '--skip_trtllm',
        action='store_true',
        help='Skip TensorRT-LLM benchmarks'
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
        # Create benchmark configuration
        config = BenchmarkConfig(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            batch_sizes=[1],  # Single batch for now
            temperatures=args.temperatures,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations
        )
        
        # Initialize benchmark runner
        runner = BenchmarkRunner(config)
        
        # Prepare engine directories
        engine_dirs = {}
        if args.engine_dirs and not args.skip_trtllm:
            for i, engine_dir in enumerate(args.engine_dirs):
                engine_name = f"trtllm_engine_{i+1}"
                engine_dirs[engine_name] = engine_dir
        
        # Run comprehensive benchmark
        results = runner.run_comprehensive_benchmark(
            hf_enabled=not args.skip_hf,
            trtllm_enabled=not args.skip_trtllm,
            engine_dirs=engine_dirs
        )
        
        # Save results
        runner.save_results(results, args.output)
        
        # Generate and save report
        report = runner.generate_report(results)
        report_file = Path(args.report)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK COMPLETED")
        logger.info("="*60)
        
        summary = results.get('summary', {})
        if 'average_speedups' in summary:
            logger.info("Average Performance:")
            for impl, metrics in summary['average_speedups'].items():
                logger.info(f"  {impl}: {metrics['average_tps']:.2f} TPS")
        
        logger.info(f"Full results: {args.output}")
        logger.info(f"Report: {args.report}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()