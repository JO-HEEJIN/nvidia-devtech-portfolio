#!/usr/bin/env python3
"""
Performance Benchmarking Suite for TensorRT Optimization

This module provides comprehensive performance comparison between PyTorch
and TensorRT models across different precision modes and batch sizes.

Benchmarking methodology:
- Warmup iterations to stabilize GPU state
- Statistical analysis of latency distribution
- Memory usage monitoring with pynvml
- Throughput calculation for production scenarios
- Results export for visualization and analysis
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pynvml
from coloredlogs import install as setup_colored_logs
from tabulate import tabulate
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import TensorRTInferenceEngine


class GPUMonitor:
    """
    GPU monitoring utility using NVIDIA Management Library (NVML).
    
    Tracks GPU memory usage, utilization, and temperature during benchmarks
    to provide comprehensive performance analysis.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU monitor.
        
        Args:
            device_id: CUDA device ID to monitor
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.device_name = pynvml.nvmlDeviceGetName(self.handle).decode('utf-8')
            self.enabled = True
            
            self.logger.info(f"GPU monitoring enabled for: {self.device_name}")
            
        except Exception as e:
            self.logger.warning(f"GPU monitoring disabled: {e}")
            self.enabled = False
            
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with memory metrics in MB
        """
        if not self.enabled:
            return {}
            
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                'total_mb': mem_info.total / 1024 / 1024,
                'used_mb': mem_info.used / 1024 / 1024,
                'free_mb': mem_info.free / 1024 / 1024,
                'utilization_percent': (mem_info.used / mem_info.total) * 100
            }
        except:
            return {}
            
    def get_utilization(self) -> Dict[str, int]:
        """
        Get GPU utilization metrics.
        
        Returns:
            Dictionary with utilization percentages
        """
        if not self.enabled:
            return {}
            
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {
                'gpu_percent': util.gpu,
                'memory_percent': util.memory
            }
        except:
            return {}
            
    def get_temperature(self) -> Optional[int]:
        """
        Get GPU temperature in Celsius.
        
        Returns:
            Temperature in Celsius or None
        """
        if not self.enabled:
            return None
            
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self.handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
        except:
            return None
            
    def cleanup(self):
        """Cleanup NVML resources."""
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class PyTorchBenchmark:
    """
    Benchmark suite for PyTorch models.
    
    Provides consistent benchmarking interface for PyTorch models
    to enable fair comparison with TensorRT optimized versions.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        use_fp16: bool = False
    ):
        """
        Initialize PyTorch benchmark.
        
        Args:
            model_name: Name of the model from torchvision
            device: Device to run on ('cuda' or 'cpu')
            use_fp16: Use FP16 precision (requires CUDA)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and device == 'cuda'
        
        # Load model
        self.model = self._load_model(model_name)
        
        # Enable FP16 if requested
        if self.use_fp16:
            self.model = self.model.half()
            self.logger.info("Using FP16 precision for PyTorch")
            
        self.logger.info(f"PyTorch model loaded: {model_name}")
        
    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load PyTorch model from torchvision.
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded model in eval mode
        """
        if hasattr(models, model_name):
            model = getattr(models, model_name)(pretrained=True)
        else:
            raise ValueError(f"Model {model_name} not found in torchvision")
            
        model = model.to(self.device)
        model.eval()
        
        return model
        
    def warmup(self, input_shape: Tuple[int, ...], iterations: int = 10):
        """
        Perform warmup iterations.
        
        Args:
            input_shape: Shape of input tensor
            iterations: Number of warmup iterations
        """
        self.logger.info(f"Running {iterations} warmup iterations...")
        
        dummy_input = torch.randn(*input_shape).to(self.device)
        if self.use_fp16:
            dummy_input = dummy_input.half()
            
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
                
        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Run benchmark and collect metrics.
        
        Args:
            input_shape: Input tensor shape
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with performance metrics
        """
        # Warmup
        if warmup_iterations > 0:
            self.warmup(input_shape, warmup_iterations)
            
        # Create input tensor
        input_tensor = torch.randn(*input_shape).to(self.device)
        if self.use_fp16:
            input_tensor = input_tensor.half()
            
        latencies = []
        
        with torch.no_grad():
            for _ in range(iterations):
                # Synchronize before timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                start = time.perf_counter()
                
                _ = self.model(input_tensor)
                
                # Synchronize after execution
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
                
        latencies = np.array(latencies)
        
        # Calculate metrics
        batch_size = input_shape[0]
        metrics = {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': float(1000.0 / np.mean(latencies) * batch_size)
        }
        
        return metrics


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for comparing PyTorch vs TensorRT.
    
    Orchestrates benchmarks across different:
    - Precision modes (FP32, FP16, INT8)
    - Batch sizes
    - Model architectures
    """
    
    def __init__(
        self,
        pytorch_model: str,
        trt_engines_dir: str,
        batch_sizes: List[int],
        input_size: Tuple[int, int] = (224, 224),
        verbose: bool = False
    ):
        """
        Initialize benchmark suite.
        
        Args:
            pytorch_model: PyTorch model name
            trt_engines_dir: Directory containing TRT engines
            batch_sizes: List of batch sizes to test
            input_size: Input image size (H, W)
            verbose: Enable verbose logging
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pytorch_model = pytorch_model
        self.trt_engines_dir = Path(trt_engines_dir)
        self.batch_sizes = batch_sizes
        self.input_size = input_size
        self.verbose = verbose
        
        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor()
        
        # Results storage
        self.results = {
            'metadata': {
                'model': pytorch_model,
                'input_size': input_size,
                'batch_sizes': batch_sizes,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'benchmarks': {}
        }
        
    def benchmark_pytorch(
        self,
        batch_size: int,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark PyTorch model.
        
        Args:
            batch_size: Batch size for inference
            iterations: Number of iterations
            warmup: Warmup iterations
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking PyTorch (batch_size={batch_size})")
        
        # Input shape (batch_size, channels, height, width)
        input_shape = (batch_size, 3, *self.input_size)
        
        # FP32 benchmark
        bench_fp32 = PyTorchBenchmark(self.pytorch_model, use_fp16=False)
        
        # Get initial memory
        gc.collect()
        torch.cuda.empty_cache()
        mem_before = self.gpu_monitor.get_memory_info()
        
        # Run benchmark
        metrics_fp32 = bench_fp32.benchmark(input_shape, iterations, warmup)
        
        # Get memory after
        mem_after = self.gpu_monitor.get_memory_info()
        
        if mem_before and mem_after:
            metrics_fp32['memory_used_mb'] = mem_after['used_mb'] - mem_before['used_mb']
        
        # FP16 benchmark (if supported)
        metrics_fp16 = None
        if torch.cuda.is_available():
            bench_fp16 = PyTorchBenchmark(self.pytorch_model, use_fp16=True)
            
            gc.collect()
            torch.cuda.empty_cache()
            mem_before = self.gpu_monitor.get_memory_info()
            
            metrics_fp16 = bench_fp16.benchmark(input_shape, iterations, warmup)
            
            mem_after = self.gpu_monitor.get_memory_info()
            if mem_before and mem_after:
                metrics_fp16['memory_used_mb'] = mem_after['used_mb'] - mem_before['used_mb']
                
        return {
            'fp32': metrics_fp32,
            'fp16': metrics_fp16
        }
        
    def benchmark_tensorrt(
        self,
        engine_path: str,
        batch_size: int,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark TensorRT engine.
        
        Args:
            engine_path: Path to TRT engine
            batch_size: Batch size
            iterations: Number of iterations
            warmup: Warmup iterations
            
        Returns:
            Benchmark results
        """
        precision = self._get_precision_from_path(engine_path)
        self.logger.info(f"Benchmarking TensorRT {precision} (batch_size={batch_size})")
        
        # Input shape
        input_shape = (batch_size, 3, *self.input_size)
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Get initial memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_before = self.gpu_monitor.get_memory_info()
        
        # Create engine and run benchmark
        with TensorRTInferenceEngine(
            engine_path,
            max_batch_size=max(self.batch_sizes),
            verbose=self.verbose
        ) as engine:
            
            metrics = engine.benchmark(dummy_input, iterations, warmup)
            
            # Get engine memory usage
            engine_memory = engine.get_memory_usage()
            metrics['engine_memory_mb'] = engine_memory['total_memory_mb']
            
        # Get memory after
        mem_after = self.gpu_monitor.get_memory_info()
        
        if mem_before and mem_after:
            metrics['memory_used_mb'] = mem_after['used_mb'] - mem_before['used_mb']
            
        return metrics
        
    def _get_precision_from_path(self, engine_path: str) -> str:
        """Extract precision mode from engine filename."""
        engine_path = str(engine_path).lower()
        if 'int8' in engine_path:
            return 'INT8'
        elif 'fp16' in engine_path:
            return 'FP16'
        else:
            return 'FP32'
            
    def find_trt_engines(self) -> Dict[str, Path]:
        """
        Find all TensorRT engines in the directory.
        
        Returns:
            Dictionary mapping precision to engine path
        """
        engines = {}
        
        for engine_path in self.trt_engines_dir.glob('*.trt'):
            precision = self._get_precision_from_path(engine_path)
            engines[precision.lower()] = engine_path
            
        self.logger.info(f"Found TensorRT engines: {list(engines.keys())}")
        return engines
        
    def run_benchmarks(
        self,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Run complete benchmark suite.
        
        Args:
            iterations: Number of iterations per benchmark
            warmup: Warmup iterations
            
        Returns:
            Complete benchmark results
        """
        self.logger.info("="*60)
        self.logger.info("Starting Benchmark Suite")
        self.logger.info("="*60)
        
        # Find TensorRT engines
        trt_engines = self.find_trt_engines()
        
        # Run benchmarks for each batch size
        for batch_size in tqdm(self.batch_sizes, desc="Batch sizes"):
            self.logger.info(f"\nBatch size: {batch_size}")
            self.logger.info("-"*40)
            
            results_batch = {}
            
            # Benchmark PyTorch
            try:
                pytorch_results = self.benchmark_pytorch(batch_size, iterations, warmup)
                results_batch['pytorch'] = pytorch_results
            except Exception as e:
                self.logger.error(f"PyTorch benchmark failed: {e}")
                results_batch['pytorch'] = {'error': str(e)}
                
            # Benchmark TensorRT engines
            for precision, engine_path in trt_engines.items():
                try:
                    trt_results = self.benchmark_tensorrt(
                        engine_path,
                        batch_size,
                        iterations,
                        warmup
                    )
                    results_batch[f'tensorrt_{precision}'] = trt_results
                except Exception as e:
                    self.logger.error(f"TensorRT {precision} benchmark failed: {e}")
                    results_batch[f'tensorrt_{precision}'] = {'error': str(e)}
                    
            self.results['benchmarks'][f'batch_{batch_size}'] = results_batch
            
        # Add GPU info to metadata
        if self.gpu_monitor.enabled:
            self.results['metadata']['gpu'] = self.gpu_monitor.device_name
            
        return self.results
        
    def print_summary(self):
        """Print benchmark summary in tabular format."""
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Prepare data for table
        table_data = []
        headers = ['Batch Size', 'Framework', 'Precision', 'Mean Latency (ms)', 
                   'Throughput (FPS)', 'Memory (MB)']
        
        for batch_key, batch_results in self.results['benchmarks'].items():
            batch_size = int(batch_key.split('_')[1])
            
            # PyTorch results
            if 'pytorch' in batch_results:
                pytorch_res = batch_results['pytorch']
                
                if 'fp32' in pytorch_res and not isinstance(pytorch_res['fp32'], str):
                    table_data.append([
                        batch_size,
                        'PyTorch',
                        'FP32',
                        f"{pytorch_res['fp32']['mean_latency_ms']:.2f}",
                        f"{pytorch_res['fp32']['throughput_fps']:.1f}",
                        f"{pytorch_res['fp32'].get('memory_used_mb', 0):.1f}"
                    ])
                    
                if 'fp16' in pytorch_res and pytorch_res['fp16']:
                    table_data.append([
                        batch_size,
                        'PyTorch',
                        'FP16',
                        f"{pytorch_res['fp16']['mean_latency_ms']:.2f}",
                        f"{pytorch_res['fp16']['throughput_fps']:.1f}",
                        f"{pytorch_res['fp16'].get('memory_used_mb', 0):.1f}"
                    ])
                    
            # TensorRT results
            for key, value in batch_results.items():
                if key.startswith('tensorrt_') and not isinstance(value, dict) or 'error' not in value:
                    precision = key.split('_')[1].upper()
                    if isinstance(value, dict) and 'mean_latency_ms' in value:
                        table_data.append([
                            batch_size,
                            'TensorRT',
                            precision,
                            f"{value['mean_latency_ms']:.2f}",
                            f"{value['throughput_fps']:.1f}",
                            f"{value.get('memory_used_mb', 0):.1f}"
                        ])
                        
        # Print table
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            
        # Print speedup summary
        print("\n" + "="*80)
        print("SPEEDUP ANALYSIS")
        print("="*80)
        
        self._print_speedup_analysis()
        
    def _print_speedup_analysis(self):
        """Calculate and print speedup metrics."""
        
        for batch_key, batch_results in self.results['benchmarks'].items():
            batch_size = int(batch_key.split('_')[1])
            
            # Get PyTorch baseline
            if 'pytorch' not in batch_results or 'fp32' not in batch_results['pytorch']:
                continue
                
            baseline = batch_results['pytorch']['fp32']
            if isinstance(baseline, str) or 'mean_latency_ms' not in baseline:
                continue
                
            baseline_latency = baseline['mean_latency_ms']
            
            print(f"\nBatch Size {batch_size}:")
            print("-" * 40)
            
            # Calculate speedups
            speedups = []
            for key, value in batch_results.items():
                if key.startswith('tensorrt_') and isinstance(value, dict) and 'mean_latency_ms' in value:
                    precision = key.split('_')[1].upper()
                    speedup = baseline_latency / value['mean_latency_ms']
                    memory_reduction = 1 - (value.get('memory_used_mb', 0) / baseline.get('memory_used_mb', 1))
                    
                    speedups.append({
                        'precision': precision,
                        'speedup': speedup,
                        'memory_reduction': memory_reduction * 100
                    })
                    
            # Sort by speedup
            speedups.sort(key=lambda x: x['speedup'], reverse=True)
            
            for item in speedups:
                print(f"  TensorRT {item['precision']}: {item['speedup']:.2f}x speedup, "
                      f"{item['memory_reduction']:.1f}% memory reduction")
                      
    def save_results(self, output_path: str):
        """
        Save benchmark results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point for benchmarking."""
    
    parser = argparse.ArgumentParser(
        description='Benchmark PyTorch vs TensorRT performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--pytorch-model',
        type=str,
        default='resnet50',
        help='PyTorch model name from torchvision'
    )
    
    parser.add_argument(
        '--trt-engines',
        type=str,
        required=True,
        help='Directory containing TensorRT engines'
    )
    
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=[1, 4, 8, 16],
        help='Batch sizes to benchmark'
    )
    
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Input image size (height width)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup iterations'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/benchmark.json',
        help='Output path for results JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_colored_logs(
        level=level,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create benchmark suite
        suite = BenchmarkSuite(
            pytorch_model=args.pytorch_model,
            trt_engines_dir=args.trt_engines,
            batch_sizes=args.batch_sizes,
            input_size=tuple(args.input_size),
            verbose=args.verbose
        )
        
        # Run benchmarks
        results = suite.run_benchmarks(
            iterations=args.iterations,
            warmup=args.warmup
        )
        
        # Print summary
        suite.print_summary()
        
        # Save results
        suite.save_results(args.output)
        
    except Exception as e:
        logging.error(f"Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'suite' in locals():
            suite.gpu_monitor.cleanup()


if __name__ == '__main__':
    main()