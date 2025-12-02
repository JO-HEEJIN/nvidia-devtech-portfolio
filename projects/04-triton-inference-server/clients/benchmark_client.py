#!/usr/bin/env python3
"""
Comprehensive benchmark client for Triton Inference Server
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
import json
import asyncio
import concurrent.futures
from datetime import datetime


class TritonBenchmark:
    """
    Benchmark suite for Triton Inference Server
    """
    
    def __init__(
        self,
        server_url_http: str = "localhost:8000",
        server_url_grpc: str = "localhost:8001",
        output_dir: str = "benchmark_results"
    ):
        """
        Initialize benchmark suite
        
        Args:
            server_url_http: HTTP server URL
            server_url_grpc: gRPC server URL
            output_dir: Output directory for results
        """
        self.server_url_http = server_url_http
        self.server_url_grpc = server_url_grpc
        self.output_dir = output_dir
        
        # Import clients
        from http_client import TritonHTTPClient
        from grpc_client import TritonGRPCClient
        import tritonclient.grpc as grpcclient
        
        self.http_client = TritonHTTPClient(server_url_http)
        self.grpc_client = TritonGRPCClient(server_url_grpc)
        self.grpcclient = grpcclient
        
        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "server": {
                "http": server_url_http,
                "grpc": server_url_grpc
            },
            "benchmarks": []
        }
    
    def benchmark_latency(
        self,
        model_name: str,
        protocol: str = "grpc",
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark inference latency
        
        Args:
            model_name: Model to benchmark
            protocol: Protocol to use (http/grpc)
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations per test
            warmup: Number of warmup iterations
        
        Returns:
            Benchmark results dictionary
        """
        print(f"\nBenchmarking {model_name} latency ({protocol})...")
        
        results = {
            "model": model_name,
            "protocol": protocol,
            "batch_results": {}
        }
        
        for batch_size in batch_sizes:
            print(f"  Batch size {batch_size}...")
            
            # Prepare dummy input based on model
            if "resnet50" in model_name:
                dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
                input_name = "input__0"
                input_shape = [batch_size, 3, 224, 224]
            elif "yolov8" in model_name:
                dummy_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
                input_name = "images"
                input_shape = [batch_size, 3, 640, 640]
            elif "bert" in model_name:
                dummy_ids = np.random.randint(0, 30000, (batch_size, 128)).astype(np.int64)
                dummy_mask = np.ones((batch_size, 128), dtype=np.int64)
            else:
                continue
            
            latencies = []
            
            # Warmup
            for _ in range(warmup):
                if protocol == "http":
                    if "bert" in model_name:
                        inputs = [
                            {
                                "name": "input_ids",
                                "shape": [batch_size, 128],
                                "datatype": "INT64",
                                "data": dummy_ids.tolist()
                            },
                            {
                                "name": "attention_mask",
                                "shape": [batch_size, 128],
                                "datatype": "INT64",
                                "data": dummy_mask.tolist()
                            }
                        ]
                    else:
                        inputs = [{
                            "name": input_name,
                            "shape": input_shape,
                            "datatype": "FP32",
                            "data": dummy_input.tolist()
                        }]
                    _ = self.http_client.infer(model_name, inputs)
                
                elif protocol == "grpc":
                    if "bert" in model_name:
                        inputs = []
                        input_ids = self.grpcclient.InferInput("input_ids", [batch_size, 128], "INT64")
                        input_ids.set_data_from_numpy(dummy_ids)
                        inputs.append(input_ids)
                        
                        attention_mask = self.grpcclient.InferInput("attention_mask", [batch_size, 128], "INT64")
                        attention_mask.set_data_from_numpy(dummy_mask)
                        inputs.append(attention_mask)
                    else:
                        inputs = []
                        input_tensor = self.grpcclient.InferInput(input_name, input_shape, "FP32")
                        input_tensor.set_data_from_numpy(dummy_input)
                        inputs.append(input_tensor)
                    _ = self.grpc_client.infer(model_name, inputs)
            
            # Benchmark
            for _ in range(iterations):
                start = time.perf_counter()
                
                if protocol == "http":
                    if "bert" in model_name:
                        inputs = [
                            {
                                "name": "input_ids",
                                "shape": [batch_size, 128],
                                "datatype": "INT64",
                                "data": dummy_ids.tolist()
                            },
                            {
                                "name": "attention_mask",
                                "shape": [batch_size, 128],
                                "datatype": "INT64",
                                "data": dummy_mask.tolist()
                            }
                        ]
                    else:
                        inputs = [{
                            "name": input_name,
                            "shape": input_shape,
                            "datatype": "FP32",
                            "data": dummy_input.tolist()
                        }]
                    _ = self.http_client.infer(model_name, inputs)
                
                elif protocol == "grpc":
                    if "bert" in model_name:
                        inputs = []
                        input_ids = self.grpcclient.InferInput("input_ids", [batch_size, 128], "INT64")
                        input_ids.set_data_from_numpy(dummy_ids)
                        inputs.append(input_ids)
                        
                        attention_mask = self.grpcclient.InferInput("attention_mask", [batch_size, 128], "INT64")
                        attention_mask.set_data_from_numpy(dummy_mask)
                        inputs.append(attention_mask)
                    else:
                        inputs = []
                        input_tensor = self.grpcclient.InferInput(input_name, input_shape, "FP32")
                        input_tensor.set_data_from_numpy(dummy_input)
                        inputs.append(input_tensor)
                    _ = self.grpc_client.infer(model_name, inputs)
                
                latencies.append((time.perf_counter() - start) * 1000)
            
            # Calculate statistics
            results["batch_results"][batch_size] = {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "throughput": batch_size / (np.mean(latencies) / 1000)
            }
            
            print(f"    Mean: {results['batch_results'][batch_size]['mean']:.2f} ms")
            print(f"    Throughput: {results['batch_results'][batch_size]['throughput']:.2f} samples/s")
        
        return results
    
    def benchmark_throughput(
        self,
        model_name: str,
        protocol: str = "grpc",
        batch_size: int = 8,
        duration: float = 30.0
    ) -> Dict:
        """
        Benchmark maximum throughput
        
        Args:
            model_name: Model to benchmark
            protocol: Protocol to use
            batch_size: Batch size for throughput test
            duration: Test duration in seconds
        
        Returns:
            Throughput results
        """
        print(f"\nBenchmarking {model_name} throughput ({protocol})...")
        print(f"  Batch size: {batch_size}")
        print(f"  Duration: {duration}s")
        
        # Prepare dummy input
        if "resnet50" in model_name:
            dummy_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            input_name = "input__0"
            input_shape = [batch_size, 3, 224, 224]
        elif "yolov8" in model_name:
            dummy_input = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
            input_name = "images"
            input_shape = [batch_size, 3, 640, 640]
        elif "bert" in model_name:
            dummy_ids = np.random.randint(0, 30000, (batch_size, 128)).astype(np.int64)
            dummy_mask = np.ones((batch_size, 128), dtype=np.int64)
        
        # Run throughput test
        start_time = time.perf_counter()
        request_count = 0
        latencies = []
        
        while (time.perf_counter() - start_time) < duration:
            req_start = time.perf_counter()
            
            if protocol == "http":
                if "bert" in model_name:
                    inputs = [
                        {
                            "name": "input_ids",
                            "shape": [batch_size, 128],
                            "datatype": "INT64",
                            "data": dummy_ids.tolist()
                        },
                        {
                            "name": "attention_mask",
                            "shape": [batch_size, 128],
                            "datatype": "INT64",
                            "data": dummy_mask.tolist()
                        }
                    ]
                else:
                    inputs = [{
                        "name": input_name,
                        "shape": input_shape,
                        "datatype": "FP32",
                        "data": dummy_input.tolist()
                    }]
                _ = self.http_client.infer(model_name, inputs)
            
            elif protocol == "grpc":
                if "bert" in model_name:
                    inputs = []
                    input_ids = self.grpcclient.InferInput("input_ids", [batch_size, 128], "INT64")
                    input_ids.set_data_from_numpy(dummy_ids)
                    inputs.append(input_ids)
                    
                    attention_mask = self.grpcclient.InferInput("attention_mask", [batch_size, 128], "INT64")
                    attention_mask.set_data_from_numpy(dummy_mask)
                    inputs.append(attention_mask)
                else:
                    inputs = []
                    input_tensor = self.grpcclient.InferInput(input_name, input_shape, "FP32")
                    input_tensor.set_data_from_numpy(dummy_input)
                    inputs.append(input_tensor)
                _ = self.grpc_client.infer(model_name, inputs)
            
            latencies.append((time.perf_counter() - req_start) * 1000)
            request_count += 1
        
        actual_duration = time.perf_counter() - start_time
        
        results = {
            "model": model_name,
            "protocol": protocol,
            "batch_size": batch_size,
            "duration": actual_duration,
            "total_requests": request_count,
            "total_samples": request_count * batch_size,
            "requests_per_second": request_count / actual_duration,
            "samples_per_second": (request_count * batch_size) / actual_duration,
            "avg_latency_ms": float(np.mean(latencies)),
            "p99_latency_ms": float(np.percentile(latencies, 99))
        }
        
        print(f"  Total requests: {results['total_requests']}")
        print(f"  Requests/second: {results['requests_per_second']:.2f}")
        print(f"  Samples/second: {results['samples_per_second']:.2f}")
        
        return results
    
    def benchmark_dynamic_batching(
        self,
        model_name: str,
        concurrent_clients: List[int] = [1, 10, 50, 100],
        requests_per_client: int = 100
    ) -> Dict:
        """
        Benchmark dynamic batching effectiveness
        
        Args:
            model_name: Model to benchmark
            concurrent_clients: List of concurrent client counts
            requests_per_client: Requests per client
        
        Returns:
            Dynamic batching results
        """
        print(f"\nBenchmarking {model_name} dynamic batching...")
        
        import asyncio
        from async_client import AsyncTritonClient
        
        results = {
            "model": model_name,
            "client_results": {}
        }
        
        async def run_concurrent_test(num_clients):
            # Prepare dummy input
            if "resnet50" in model_name:
                dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                inputs = [{
                    "name": "input__0",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": dummy_input.tolist()
                }]
            elif "yolov8" in model_name:
                dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
                inputs = [{
                    "name": "images",
                    "shape": [1, 3, 640, 640],
                    "datatype": "FP32",
                    "data": dummy_input.tolist()
                }]
            elif "bert" in model_name:
                dummy_ids = np.random.randint(0, 30000, (1, 128)).astype(np.int64)
                dummy_mask = np.ones((1, 128), dtype=np.int64)
                inputs = [
                    {
                        "name": "input_ids",
                        "shape": [1, 128],
                        "datatype": "INT64",
                        "data": dummy_ids.tolist()
                    },
                    {
                        "name": "attention_mask",
                        "shape": [1, 128],
                        "datatype": "INT64",
                        "data": dummy_mask.tolist()
                    }
                ]
            
            async with AsyncTritonClient(
                self.server_url_http,
                num_clients
            ) as client:
                tasks = []
                for i in range(num_clients * requests_per_client):
                    task = client.infer(
                        model_name=model_name,
                        inputs=inputs,
                        request_id=f"batch_{i}"
                    )
                    tasks.append(task)
                
                start_time = time.perf_counter()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.perf_counter() - start_time
                
                # Analyze results
                successful = [r for r in results if not isinstance(r, Exception)]
                latencies = [r["_client_latency_ms"] for r in successful]
                
                return {
                    "total_time": total_time,
                    "successful_requests": len(successful),
                    "failed_requests": len(results) - len(successful),
                    "throughput": len(successful) / total_time,
                    "avg_latency": float(np.mean(latencies)) if latencies else 0,
                    "p99_latency": float(np.percentile(latencies, 99)) if latencies else 0
                }
        
        for num_clients in concurrent_clients:
            print(f"  Testing with {num_clients} concurrent clients...")
            
            result = asyncio.run(run_concurrent_test(num_clients))
            results["client_results"][num_clients] = result
            
            print(f"    Throughput: {result['throughput']:.2f} QPS")
            print(f"    Avg latency: {result['avg_latency']:.2f} ms")
        
        return results
    
    def generate_report(self):
        """Generate benchmark report with visualizations"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{self.output_dir}/benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}/benchmark_results.json")
    
    def run_full_benchmark(self, models: List[str]):
        """
        Run complete benchmark suite
        
        Args:
            models: List of model names to benchmark
        """
        print("="*60)
        print("Starting Triton Benchmark Suite")
        print("="*60)
        
        for model in models:
            # Check if model is ready
            if not self.http_client.is_model_ready(model):
                print(f"Skipping {model} - not ready")
                continue
            
            # Latency benchmark - HTTP
            result = self.benchmark_latency(
                model, "http",
                batch_sizes=[1, 4, 8, 16],
                iterations=50
            )
            self.results["benchmarks"].append(result)
            
            # Latency benchmark - gRPC
            result = self.benchmark_latency(
                model, "grpc",
                batch_sizes=[1, 4, 8, 16],
                iterations=50
            )
            self.results["benchmarks"].append(result)
            
            # Throughput benchmark
            result = self.benchmark_throughput(
                model, "grpc",
                batch_size=8,
                duration=10.0
            )
            self.results["benchmarks"].append(result)
            
            # Dynamic batching benchmark
            result = self.benchmark_dynamic_batching(
                model,
                concurrent_clients=[1, 10, 50],
                requests_per_client=50
            )
            self.results["benchmarks"].append(result)
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("Benchmark Complete!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Triton Benchmark Client')
    parser.add_argument('--http-server', type=str, default='localhost:8000',
                        help='HTTP server URL')
    parser.add_argument('--grpc-server', type=str, default='localhost:8001',
                        help='gRPC server URL')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['resnet50_pytorch', 'yolov8_tensorrt', 'bert_onnx'],
                        help='Models to benchmark')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = TritonBenchmark(
        args.http_server,
        args.grpc_server,
        args.output_dir
    )
    
    # Run benchmarks
    benchmark.run_full_benchmark(args.models)


if __name__ == '__main__':
    main()