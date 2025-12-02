#!/usr/bin/env python3
"""
Benchmark YOLOv8 inference performance
"""

import argparse
import json
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import warnings

# Handle optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. PyTorch benchmarking disabled.")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    warnings.warn("TensorRT not available. TensorRT benchmarking disabled.")

from inference_pytorch import PyTorchInference
from inference_tensorrt import TensorRTInference


class Benchmarker:
    """
    Comprehensive benchmarking for YOLOv8 inference
    """
    
    def __init__(
        self,
        pytorch_model: str = None,
        tensorrt_engine: str = None,
        input_size: int = 640,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        iterations: int = 100,
        warmup: int = 10
    ):
        """
        Initialize benchmarker
        
        Args:
            pytorch_model: Path to PyTorch model
            tensorrt_engine: Path to TensorRT engine
            input_size: Input image size
            batch_sizes: Batch sizes to test
            iterations: Number of iterations per test
            warmup: Warmup iterations
        """
        self.pytorch_model = pytorch_model
        self.tensorrt_engine = tensorrt_engine
        self.input_size = input_size
        self.batch_sizes = batch_sizes
        self.iterations = iterations
        self.warmup = warmup
        
        # Results storage
        self.results = {
            'pytorch': {},
            'tensorrt': {},
            'metadata': {
                'input_size': input_size,
                'iterations': iterations,
                'warmup': warmup
            }
        }
        
        # Setup engines
        self.setup_engines()
    
    def setup_engines(self):
        """
        Initialize inference engines
        """
        self.engines = {}
        
        if self.pytorch_model and TORCH_AVAILABLE:
            print(f"Loading PyTorch model: {self.pytorch_model}")
            self.engines['pytorch'] = PyTorchInference(
                self.pytorch_model,
                input_size=(self.input_size, self.input_size)
            )
        
        if self.tensorrt_engine and TRT_AVAILABLE:
            print(f"Loading TensorRT engine: {self.tensorrt_engine}")
            self.engines['tensorrt'] = TensorRTInference(
                self.tensorrt_engine,
                input_size=(self.input_size, self.input_size)
            )
    
    def benchmark_latency(
        self,
        engine,
        batch_size: int
    ) -> Dict:
        """
        Benchmark inference latency
        
        Args:
            engine: Inference engine
            batch_size: Batch size
        
        Returns:
            Latency statistics
        """
        # Create dummy input
        if batch_size == 1:
            dummy = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            dummy = [np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8) 
                    for _ in range(batch_size)]
        
        # Warmup
        for _ in range(self.warmup):
            if batch_size == 1:
                _ = engine.infer(dummy)
            else:
                _ = engine.infer_batch(dummy)
        
        # Measure
        latencies = []
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            
            if batch_size == 1:
                _ = engine.infer(dummy)
            else:
                _ = engine.infer_batch(dummy)
            
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'min': float(np.min(latencies)),
            'max': float(np.max(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
            'fps': float(1000 / np.mean(latencies) * batch_size),
            'throughput': float(batch_size * 1000 / np.mean(latencies))
        }
    
    def benchmark_memory(self) -> Dict:
        """
        Benchmark memory usage
        
        Returns:
            Memory statistics
        """
        memory_stats = {}
        
        # System memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_stats['cpu_memory_mb'] = memory_info.rss / 1024 / 1024
        
        # GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_stats['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_stats
    
    def run_benchmarks(self):
        """
        Run all benchmarks
        """
        print("\n" + "="*60)
        print("Starting Benchmark Suite")
        print("="*60)
        
        for engine_name, engine in self.engines.items():
            print(f"\nBenchmarking {engine_name.upper()}")
            print("-"*40)
            
            self.results[engine_name]['memory'] = self.benchmark_memory()
            self.results[engine_name]['latency'] = {}
            
            for batch_size in self.batch_sizes:
                if batch_size > 1 and engine_name == 'pytorch':
                    # Skip large batches for PyTorch if slow
                    if batch_size > 8:
                        continue
                
                print(f"  Batch size {batch_size}...")
                
                try:
                    stats = self.benchmark_latency(engine, batch_size)
                    self.results[engine_name]['latency'][batch_size] = stats
                    
                    print(f"    Mean: {stats['mean']:.2f} ms")
                    print(f"    FPS: {stats['fps']:.1f}")
                    print(f"    P95: {stats['p95']:.2f} ms")
                except Exception as e:
                    print(f"    Error: {e}")
                    self.results[engine_name]['latency'][batch_size] = None
    
    def calculate_speedup(self):
        """
        Calculate TensorRT speedup over PyTorch
        """
        if 'pytorch' not in self.results or 'tensorrt' not in self.results:
            return
        
        self.results['speedup'] = {}
        
        for batch_size in self.batch_sizes:
            pt_stats = self.results['pytorch']['latency'].get(batch_size)
            trt_stats = self.results['tensorrt']['latency'].get(batch_size)
            
            if pt_stats and trt_stats:
                self.results['speedup'][batch_size] = {
                    'latency': pt_stats['mean'] / trt_stats['mean'],
                    'throughput': trt_stats['throughput'] / pt_stats['throughput']
                }
    
    def save_results(self, output_path: str = 'benchmark_results.json'):
        """
        Save benchmark results to JSON
        """
        # Add system info
        self.results['metadata']['system'] = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.results['metadata']['gpu'] = {
                'name': torch.cuda.get_device_name(),
                'capability': f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
            }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def plot_results(self, output_dir: str = 'results'):
        """
        Generate performance charts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup style
        sns.set_style("whitegrid")
        
        # 1. Latency comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean latency
        ax = axes[0, 0]
        for engine_name in ['pytorch', 'tensorrt']:
            if engine_name in self.results:
                batch_sizes = []
                latencies = []
                
                for bs, stats in self.results[engine_name]['latency'].items():
                    if stats:
                        batch_sizes.append(bs)
                        latencies.append(stats['mean'])
                
                if batch_sizes:
                    ax.plot(batch_sizes, latencies, 'o-', label=engine_name.upper(), linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Mean Latency Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # FPS comparison
        ax = axes[0, 1]
        for engine_name in ['pytorch', 'tensorrt']:
            if engine_name in self.results:
                batch_sizes = []
                fps_values = []
                
                for bs, stats in self.results[engine_name]['latency'].items():
                    if stats:
                        batch_sizes.append(bs)
                        fps_values.append(stats['fps'])
                
                if batch_sizes:
                    ax.plot(batch_sizes, fps_values, 'o-', label=engine_name.upper(), linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('FPS')
        ax.set_title('FPS Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Speedup
        ax = axes[1, 0]
        if 'speedup' in self.results:
            batch_sizes = []
            speedups = []
            
            for bs, sp in self.results['speedup'].items():
                batch_sizes.append(bs)
                speedups.append(sp['latency'])
            
            if batch_sizes:
                ax.bar(range(len(batch_sizes)), speedups, color='green', alpha=0.7)
                ax.set_xticks(range(len(batch_sizes)))
                ax.set_xticklabels([f'BS={bs}' for bs in batch_sizes])
                ax.set_ylabel('Speedup Factor')
                ax.set_title('TensorRT Speedup over PyTorch')
                ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                
                for i, v in enumerate(speedups):
                    ax.text(i, v + 0.1, f'{v:.1f}x', ha='center')
        
        # Latency percentiles
        ax = axes[1, 1]
        if 'tensorrt' in self.results and 1 in self.results['tensorrt']['latency']:
            stats = self.results['tensorrt']['latency'][1]
            if stats:
                percentiles = ['p50', 'p95', 'p99']
                values = [stats[p] for p in percentiles]
                
                bars = ax.bar(percentiles, values, color=['blue', 'orange', 'red'], alpha=0.7)
                ax.set_ylabel('Latency (ms)')
                ax.set_title('TensorRT Latency Percentiles (BS=1)')
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, 
                           f'{val:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Charts saved to {output_dir}")
    
    def print_summary(self):
        """
        Print benchmark summary
        """
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Performance table
        print("\nPerformance Comparison:")
        print("-"*40)
        
        headers = ["Metric", "PyTorch", "TensorRT", "Speedup"]
        rows = []
        
        for batch_size in [1, 4, 8]:
            pt_stats = self.results.get('pytorch', {}).get('latency', {}).get(batch_size)
            trt_stats = self.results.get('tensorrt', {}).get('latency', {}).get(batch_size)
            
            if pt_stats and trt_stats:
                speedup = pt_stats['mean'] / trt_stats['mean']
                rows.append([
                    f"BS={batch_size} Latency",
                    f"{pt_stats['mean']:.2f} ms",
                    f"{trt_stats['mean']:.2f} ms",
                    f"{speedup:.2f}x"
                ])
                rows.append([
                    f"BS={batch_size} FPS",
                    f"{pt_stats['fps']:.1f}",
                    f"{trt_stats['fps']:.1f}",
                    f"{trt_stats['fps']/pt_stats['fps']:.2f}x"
                ])
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print headers
        header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_str)
        print("-" * len(header_str))
        
        # Print rows
        for row in rows:
            print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
        
        # Memory usage
        print("\nMemory Usage:")
        print("-"*40)
        
        for engine_name in ['pytorch', 'tensorrt']:
            if engine_name in self.results and 'memory' in self.results[engine_name]:
                mem = self.results[engine_name]['memory']
                print(f"{engine_name.upper()}:")
                print(f"  CPU Memory: {mem.get('cpu_memory_mb', 0):.1f} MB")
                if 'gpu_memory_mb' in mem:
                    print(f"  GPU Memory: {mem.get('gpu_memory_mb', 0):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLOv8 inference')
    parser.add_argument('--pytorch-model', type=str, default='yolov8s.pt',
                        help='Path to PyTorch model')
    parser.add_argument('--tensorrt-engine', type=str, default='yolov8s.engine',
                        help='Path to TensorRT engine')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16],
                        help='Batch sizes to test')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    
    args = parser.parse_args()
    
    # Create benchmarker
    benchmarker = Benchmarker(
        pytorch_model=args.pytorch_model if Path(args.pytorch_model).exists() else None,
        tensorrt_engine=args.tensorrt_engine if Path(args.tensorrt_engine).exists() else None,
        input_size=args.input_size,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Run benchmarks
    benchmarker.run_benchmarks()
    benchmarker.calculate_speedup()
    
    # Save results
    benchmarker.save_results(args.output)
    
    # Print summary
    benchmarker.print_summary()
    
    # Generate plots
    if args.plot:
        benchmarker.plot_results()


if __name__ == '__main__':
    main()