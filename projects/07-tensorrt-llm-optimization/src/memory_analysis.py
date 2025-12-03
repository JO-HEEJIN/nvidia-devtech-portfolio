#!/usr/bin/env python3
"""
Memory analysis tools for TensorRT-LLM optimization.
Analyzes KV cache memory usage, compares paged attention vs standard attention,
and generates memory efficiency reports.
"""

import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
except ImportError as e:
    logger.error(f"Required dependencies not installed: {e}")
    logger.error("Please run: pip install torch numpy")
    exit(1)

# Try to import memory monitoring tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available. Memory monitoring will be limited.")
    PSUTIL_AVAILABLE = False

@dataclass
class MemoryConfig:
    """Configuration for memory analysis."""
    sequence_lengths: List[int]
    batch_sizes: List[int]
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int

@dataclass 
class MemoryMeasurement:
    """Single memory measurement."""
    label: str
    timestamp: float
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    gpu_memory_cached: float
    cpu_memory_mb: float
    cpu_memory_percent: float

class KVCacheAnalyzer:
    """Analyzer for KV cache memory usage patterns."""
    
    def __init__(self, config: MemoryConfig):
        """Initialize KV cache analyzer."""
        self.config = config
        self.measurements = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def calculate_kv_cache_size(
        self, 
        batch_size: int, 
        sequence_length: int, 
        use_fp16: bool = True
    ) -> Dict[str, float]:
        """Calculate theoretical KV cache memory requirements."""
        
        # Data type size in bytes
        dtype_size = 2 if use_fp16 else 4
        
        # KV cache dimensions: [batch_size, num_heads, sequence_length, head_dim]
        # We have both K and V caches, and multiple layers
        kv_cache_elements = (
            2 *  # K and V caches
            self.config.num_layers *
            batch_size *
            self.config.num_heads * 
            sequence_length *
            self.config.head_dim
        )
        
        kv_cache_bytes = kv_cache_elements * dtype_size
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        kv_cache_gb = kv_cache_mb / 1024
        
        # Additional calculations
        per_token_bytes = (
            2 *  # K and V
            self.config.num_layers *
            self.config.num_heads *
            self.config.head_dim *
            dtype_size
        )
        
        per_token_mb = per_token_bytes / (1024 * 1024)
        
        return {
            'kv_cache_mb': kv_cache_mb,
            'kv_cache_gb': kv_cache_gb,
            'per_token_mb': per_token_mb,
            'total_elements': kv_cache_elements,
            'dtype_size_bytes': dtype_size,
            'sequence_length': sequence_length,
            'batch_size': batch_size
        }
    
    def calculate_paged_attention_memory(
        self, 
        batch_size: int, 
        sequence_length: int,
        block_size: int = 64,
        use_fp16: bool = True
    ) -> Dict[str, float]:
        """Calculate memory usage for paged attention."""
        
        # Standard KV cache calculation
        standard_memory = self.calculate_kv_cache_size(batch_size, sequence_length, use_fp16)
        
        # Paged attention calculations
        dtype_size = 2 if use_fp16 else 4
        
        # Number of blocks needed per sequence
        blocks_per_sequence = math.ceil(sequence_length / block_size)
        total_blocks = batch_size * blocks_per_sequence
        
        # Each block stores: block_size tokens * num_layers * num_heads * head_dim * 2 (K+V)
        elements_per_block = (
            block_size *
            self.config.num_layers *
            self.config.num_heads *
            self.config.head_dim *
            2  # K and V
        )
        
        paged_memory_bytes = total_blocks * elements_per_block * dtype_size
        paged_memory_mb = paged_memory_bytes / (1024 * 1024)
        
        # Memory overhead due to block fragmentation
        # Last block might not be fully utilized
        wasted_tokens_per_sequence = block_size - (sequence_length % block_size)
        if wasted_tokens_per_sequence == block_size:
            wasted_tokens_per_sequence = 0
        
        total_wasted_tokens = batch_size * wasted_tokens_per_sequence
        wasted_memory_bytes = total_wasted_tokens * (elements_per_block // block_size) * dtype_size
        wasted_memory_mb = wasted_memory_bytes / (1024 * 1024)
        
        # Memory efficiency
        efficiency = (sequence_length * batch_size) / (total_blocks * block_size)
        
        return {
            'paged_memory_mb': paged_memory_mb,
            'standard_memory_mb': standard_memory['kv_cache_mb'],
            'memory_overhead_mb': paged_memory_mb - standard_memory['kv_cache_mb'],
            'memory_overhead_percent': ((paged_memory_mb / standard_memory['kv_cache_mb']) - 1) * 100,
            'wasted_memory_mb': wasted_memory_mb,
            'efficiency': efficiency,
            'blocks_per_sequence': blocks_per_sequence,
            'total_blocks': total_blocks,
            'block_size': block_size,
            'wasted_tokens': total_wasted_tokens
        }
    
    def record_memory_measurement(self, label: str) -> MemoryMeasurement:
        """Record current memory usage."""
        timestamp = time.time()
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)   # MB
            gpu_cached = torch.cuda.memory_cached() / (1024 * 1024)       # MB
        else:
            gpu_allocated = gpu_reserved = gpu_cached = 0.0
        
        # CPU memory (if psutil available)
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_memory_percent = process.memory_percent()
        else:
            cpu_memory_mb = cpu_memory_percent = 0.0
        
        measurement = MemoryMeasurement(
            label=label,
            timestamp=timestamp,
            gpu_memory_allocated=gpu_allocated,
            gpu_memory_reserved=gpu_reserved,
            gpu_memory_cached=gpu_cached,
            cpu_memory_mb=cpu_memory_mb,
            cpu_memory_percent=cpu_memory_percent
        )
        
        self.measurements.append(measurement)
        logger.debug(f"Memory [{label}]: GPU={gpu_allocated:.1f}MB, CPU={cpu_memory_mb:.1f}MB")
        
        return measurement
    
    def simulate_kv_cache_growth(
        self, 
        max_sequence_length: int, 
        batch_size: int = 1,
        use_fp16: bool = True
    ) -> Dict[str, List[float]]:
        """Simulate KV cache memory growth during generation."""
        
        sequence_lengths = range(1, max_sequence_length + 1, max(1, max_sequence_length // 50))
        standard_memory = []
        paged_memory_64 = []
        paged_memory_128 = []
        paged_memory_256 = []
        
        for seq_len in sequence_lengths:
            # Standard KV cache
            standard = self.calculate_kv_cache_size(batch_size, seq_len, use_fp16)
            standard_memory.append(standard['kv_cache_mb'])
            
            # Paged attention with different block sizes
            paged_64 = self.calculate_paged_attention_memory(batch_size, seq_len, 64, use_fp16)
            paged_memory_64.append(paged_64['paged_memory_mb'])
            
            paged_128 = self.calculate_paged_attention_memory(batch_size, seq_len, 128, use_fp16)
            paged_memory_128.append(paged_128['paged_memory_mb'])
            
            paged_256 = self.calculate_paged_attention_memory(batch_size, seq_len, 256, use_fp16)
            paged_memory_256.append(paged_256['paged_memory_mb'])
        
        return {
            'sequence_lengths': list(sequence_lengths),
            'standard_memory': standard_memory,
            'paged_memory_64': paged_memory_64,
            'paged_memory_128': paged_memory_128,
            'paged_memory_256': paged_memory_256
        }
    
    def analyze_batch_size_impact(self, max_batch_size: int = 32, sequence_length: int = 1024) -> Dict[str, Any]:
        """Analyze memory usage impact of different batch sizes."""
        
        batch_sizes = [1, 2, 4, 8, 16, max_batch_size]
        results = {
            'batch_sizes': batch_sizes,
            'standard_memory': [],
            'paged_memory': [],
            'memory_per_sequence': [],
            'efficiency': []
        }
        
        for batch_size in batch_sizes:
            # Standard memory
            standard = self.calculate_kv_cache_size(batch_size, sequence_length)
            results['standard_memory'].append(standard['kv_cache_mb'])
            results['memory_per_sequence'].append(standard['kv_cache_mb'] / batch_size)
            
            # Paged memory (using 64-token blocks)
            paged = self.calculate_paged_attention_memory(batch_size, sequence_length, 64)
            results['paged_memory'].append(paged['paged_memory_mb'])
            results['efficiency'].append(paged['efficiency'])
        
        return results
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory analysis report."""
        
        report = {
            'model_config': {
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'head_dim': self.config.head_dim,
                'vocab_size': self.config.vocab_size
            },
            'analysis_results': {},
            'recommendations': []
        }
        
        # Analyze different scenarios
        logger.info("Analyzing KV cache memory patterns...")
        
        # Scenario 1: Sequence length impact
        growth_analysis = self.simulate_kv_cache_growth(2048, batch_size=1)
        report['analysis_results']['sequence_length_impact'] = growth_analysis
        
        # Scenario 2: Batch size impact
        batch_analysis = self.analyze_batch_size_impact(32, 1024)
        report['analysis_results']['batch_size_impact'] = batch_analysis
        
        # Scenario 3: Block size comparison for paged attention
        block_sizes = [16, 32, 64, 128, 256]
        block_comparison = {}
        
        for block_size in block_sizes:
            paged_result = self.calculate_paged_attention_memory(8, 1024, block_size)
            block_comparison[f'block_{block_size}'] = {
                'memory_mb': paged_result['paged_memory_mb'],
                'efficiency': paged_result['efficiency'],
                'overhead_percent': paged_result['memory_overhead_percent'],
                'wasted_memory_mb': paged_result['wasted_memory_mb']
            }
        
        report['analysis_results']['block_size_comparison'] = block_comparison
        
        # Generate recommendations
        report['recommendations'] = self._generate_memory_recommendations(report['analysis_results'])
        
        return report
    
    def _generate_memory_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations based on analysis."""
        recommendations = []
        
        # Block size recommendations
        block_comparison = analysis_results.get('block_size_comparison', {})
        if block_comparison:
            best_block = min(block_comparison.keys(), 
                           key=lambda x: block_comparison[x]['overhead_percent'])
            best_overhead = block_comparison[best_block]['overhead_percent']
            recommendations.append(
                f"Optimal block size for paged attention: {best_block.split('_')[1]} tokens "
                f"(overhead: {best_overhead:.1f}%)"
            )
        
        # Memory efficiency recommendations
        recommendations.extend([
            "Use FP16 precision to halve KV cache memory usage compared to FP32",
            "Consider INT8 KV cache quantization for 4x memory reduction",
            "Paged attention reduces memory fragmentation for variable-length sequences",
            "For fixed batch sizes, standard attention may be more efficient than paged",
            "Monitor actual GPU memory usage during inference to validate theoretical calculations"
        ])
        
        return recommendations
    
    def save_analysis(self, results: Dict[str, Any], output_path: str):
        """Save memory analysis results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results['metadata'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'analysis_script_version': '1.0'
        }
        
        if torch.cuda.is_available():
            results['metadata']['gpu_info'] = {
                'device_name': torch.cuda.get_device_name(),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'compute_capability': f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Memory analysis saved to: {output_file}")

class MemoryProfiler:
    """Memory profiler for runtime memory usage tracking."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.measurements = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def start_profiling(self):
        """Start memory profiling session."""
        logger.info("Starting memory profiling...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        self.measurements = []
        self.record_measurement("profiling_start")
    
    def record_measurement(self, label: str):
        """Record memory measurement with label."""
        timestamp = time.time()
        
        measurement = {
            'label': label,
            'timestamp': timestamp,
        }
        
        if torch.cuda.is_available():
            measurement.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_memory_cached_mb': torch.cuda.memory_cached() / (1024 * 1024),
                'gpu_max_memory_allocated_mb': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'gpu_max_memory_reserved_mb': torch.cuda.max_memory_reserved() / (1024 * 1024),
            })
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            measurement.update({
                'cpu_memory_mb': process.memory_info().rss / (1024 * 1024),
                'cpu_memory_percent': process.memory_percent(),
            })
        
        self.measurements.append(measurement)
        logger.debug(f"Memory [{label}]: {measurement}")
    
    def get_peak_memory_usage(self) -> Dict[str, float]:
        """Get peak memory usage from all measurements."""
        if not self.measurements:
            return {}
        
        peak_usage = {}
        
        # Find peak values for each metric
        for key in self.measurements[0].keys():
            if key in ['label', 'timestamp']:
                continue
            
            values = [m.get(key, 0) for m in self.measurements]
            peak_usage[f'peak_{key}'] = max(values) if values else 0
        
        return peak_usage
    
    def generate_profile_report(self) -> Dict[str, Any]:
        """Generate memory profiling report."""
        if not self.measurements:
            return {'error': 'No measurements recorded'}
        
        peak_usage = self.get_peak_memory_usage()
        
        # Calculate memory growth
        start_measurement = self.measurements[0]
        end_measurement = self.measurements[-1]
        
        memory_growth = {}
        for key in start_measurement.keys():
            if key.endswith('_mb') and key in end_measurement:
                growth = end_measurement[key] - start_measurement[key]
                memory_growth[f'{key}_growth'] = growth
        
        report = {
            'profiling_summary': {
                'num_measurements': len(self.measurements),
                'profiling_duration_seconds': end_measurement['timestamp'] - start_measurement['timestamp'],
                'peak_memory_usage': peak_usage,
                'memory_growth': memory_growth
            },
            'measurements': self.measurements,
            'recommendations': self._generate_profiling_recommendations(peak_usage, memory_growth)
        }
        
        return report
    
    def _generate_profiling_recommendations(self, peak_usage: Dict[str, float], memory_growth: Dict[str, float]) -> List[str]:
        """Generate recommendations based on profiling results."""
        recommendations = []
        
        # GPU memory recommendations
        if 'peak_gpu_memory_allocated_mb' in peak_usage:
            peak_gpu = peak_usage['peak_gpu_memory_allocated_mb']
            if peak_gpu > 8192:  # > 8GB
                recommendations.append(f"High GPU memory usage detected ({peak_gpu:.0f}MB). Consider model quantization.")
            
            # Memory growth analysis
            gpu_growth = memory_growth.get('gpu_memory_allocated_mb_growth', 0)
            if gpu_growth > 1024:  # > 1GB growth
                recommendations.append(f"Significant GPU memory growth during inference (+{gpu_growth:.0f}MB). Check for memory leaks.")
        
        # CPU memory recommendations  
        if 'peak_cpu_memory_mb' in peak_usage:
            peak_cpu = peak_usage['peak_cpu_memory_mb']
            if peak_cpu > 4096:  # > 4GB
                recommendations.append(f"High CPU memory usage detected ({peak_cpu:.0f}MB).")
        
        # General recommendations
        recommendations.extend([
            "Use torch.cuda.empty_cache() periodically to free unused GPU memory",
            "Monitor memory usage during long inference sessions",
            "Consider gradient checkpointing for training to reduce memory usage"
        ])
        
        return recommendations

def load_model_config(config_path: str) -> MemoryConfig:
    """Load model configuration for memory analysis."""
    try:
        if config_path.endswith('.yaml'):
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            model_config = config_data.get('model', {})
            return MemoryConfig(
                sequence_lengths=[512, 1024, 2048],
                batch_sizes=[1, 2, 4, 8, 16],
                hidden_size=model_config.get('hidden_size', 2048),
                num_layers=model_config.get('num_layers', 22),
                num_heads=model_config.get('num_heads', 32),
                head_dim=model_config.get('hidden_size', 2048) // model_config.get('num_heads', 32),
                vocab_size=model_config.get('vocab_size', 32000)
            )
        else:
            # Default TinyLlama config
            return MemoryConfig(
                sequence_lengths=[512, 1024, 2048],
                batch_sizes=[1, 2, 4, 8, 16],
                hidden_size=2048,
                num_layers=22,
                num_heads=32,
                head_dim=64,
                vocab_size=32000
            )
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        logger.info("Using default TinyLlama configuration")
        return MemoryConfig(
            sequence_lengths=[512, 1024, 2048],
            batch_sizes=[1, 2, 4, 8, 16],
            hidden_size=2048,
            num_layers=22,
            num_heads=32,
            head_dim=64,
            vocab_size=32000
        )

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Memory analysis for TensorRT-LLM optimization"
    )
    parser.add_argument(
        '--config',
        default='configs/tinyllama_fp16.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--output',
        default='results/memory_analysis.json',
        help='Output file for analysis results'
    )
    parser.add_argument(
        '--max_sequence_length',
        type=int,
        default=2048,
        help='Maximum sequence length to analyze'
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=32,
        help='Maximum batch size to analyze'
    )
    parser.add_argument(
        '--profile_runtime',
        action='store_true',
        help='Profile runtime memory usage'
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
        # Load model configuration
        config = load_model_config(args.config)
        
        # Initialize analyzer
        analyzer = KVCacheAnalyzer(config)
        
        if args.profile_runtime:
            # Runtime profiling mode
            profiler = MemoryProfiler()
            profiler.start_profiling()
            
            # Simulate some memory operations
            logger.info("Simulating inference memory patterns...")
            profiler.record_measurement("before_model_load")
            
            # Simulate model loading
            if torch.cuda.is_available():
                dummy_tensor = torch.randn(config.hidden_size, config.vocab_size, device='cuda')
                profiler.record_measurement("after_model_load")
                
                # Simulate KV cache allocation
                kv_cache = torch.randn(
                    config.num_layers, 2, 4, 1024, config.head_dim, 
                    device='cuda', dtype=torch.float16
                )
                profiler.record_measurement("after_kv_cache_alloc")
                
                # Cleanup
                del dummy_tensor, kv_cache
                torch.cuda.empty_cache()
                profiler.record_measurement("after_cleanup")
            
            # Generate profiling report
            profile_results = profiler.generate_profile_report()
            
            # Save profiling results
            profile_output = args.output.replace('.json', '_profile.json')
            analyzer.save_analysis(profile_results, profile_output)
            
            logger.info(f"Runtime profiling completed. Results saved to: {profile_output}")
        
        else:
            # Theoretical analysis mode
            logger.info("Running theoretical memory analysis...")
            results = analyzer.generate_memory_report()
            
            # Save analysis results
            analyzer.save_analysis(results, args.output)
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("MEMORY ANALYSIS SUMMARY")
            logger.info("="*60)
            
            if 'recommendations' in results:
                logger.info("Key Recommendations:")
                for rec in results['recommendations']:
                    logger.info(f"  - {rec}")
            
            # Print memory usage examples
            logger.info("\nExample Memory Usage (FP16):")
            example_result = analyzer.calculate_kv_cache_size(8, 1024, True)
            logger.info(f"  Batch=8, Seq=1024: {example_result['kv_cache_mb']:.1f} MB")
            
            paged_result = analyzer.calculate_paged_attention_memory(8, 1024, 64, True)
            logger.info(f"  Paged Attention: {paged_result['paged_memory_mb']:.1f} MB (overhead: {paged_result['memory_overhead_percent']:.1f}%)")
            
            logger.info(f"\nFull analysis saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Memory analysis failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()