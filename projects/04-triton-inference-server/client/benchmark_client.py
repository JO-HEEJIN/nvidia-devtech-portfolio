#!/usr/bin/env python3
"""
Comprehensive benchmark client for Triton Inference Server
Tests different concurrency levels and generates performance report
"""

import numpy as np
import time
import argparse
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from PIL import Image


def preprocess_image(image_path):
    """Preprocess image for ResNet50 inference"""
    img = Image.open(image_path)
    img = img.resize((256, 256))
    
    # Center crop to 224x224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to CHW format and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)


class TritonBenchmark:
    """Benchmark suite for Triton Inference Server"""
    
    def __init__(self, server_url_http: str, server_url_grpc: str):
        self.server_url_http = server_url_http
        self.server_url_grpc = server_url_grpc
        self.results = []
        
    def benchmark_http(
        self,
        model_name: str,
        input_data: np.ndarray,
        num_requests: int,
        concurrency: int
    ) -> Dict:
        """Benchmark HTTP endpoint"""
        
        client = httpclient.InferenceServerClient(url=self.server_url_http)
        
        def send_request(request_id):
            inputs = []
            inputs.append(httpclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
            inputs[0].set_data_from_numpy(input_data)
            
            outputs = []
            outputs.append(httpclient.InferRequestedOutput('output__0'))
            
            start_time = time.perf_counter()
            try:
                response = client.infer(model_name, inputs, outputs=outputs)
                latency = (time.perf_counter() - start_time) * 1000
                return {"success": True, "latency_ms": latency}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Run benchmark
        latencies = []
        errors = 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    latencies.append(result["latency_ms"])
                else:
                    errors += 1
        
        total_time = time.perf_counter() - start_time
        
        if latencies:
            return {
                "protocol": "HTTP",
                "concurrency": concurrency,
                "total_requests": num_requests,
                "successful": len(latencies),
                "failed": errors,
                "total_time_s": total_time,
                "throughput_qps": len(latencies) / total_time,
                "latency_mean_ms": np.mean(latencies),
                "latency_median_ms": np.median(latencies),
                "latency_p50_ms": np.percentile(latencies, 50),
                "latency_p95_ms": np.percentile(latencies, 95),
                "latency_p99_ms": np.percentile(latencies, 99),
                "latency_min_ms": np.min(latencies),
                "latency_max_ms": np.max(latencies)
            }
        else:
            return {"protocol": "HTTP", "error": "All requests failed"}
    
    def benchmark_grpc(
        self,
        model_name: str,
        input_data: np.ndarray,
        num_requests: int,
        concurrency: int
    ) -> Dict:
        """Benchmark gRPC endpoint"""
        
        client = grpcclient.InferenceServerClient(url=self.server_url_grpc)
        
        def send_request(request_id):
            inputs = []
            inputs.append(grpcclient.InferInput('input__0', [1, 3, 224, 224], "FP32"))
            inputs[0].set_data_from_numpy(input_data)
            
            outputs = []
            outputs.append(grpcclient.InferRequestedOutput('output__0'))
            
            start_time = time.perf_counter()
            try:
                response = client.infer(model_name, inputs, outputs=outputs)
                latency = (time.perf_counter() - start_time) * 1000
                return {"success": True, "latency_ms": latency}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Run benchmark
        latencies = []
        errors = 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    latencies.append(result["latency_ms"])
                else:
                    errors += 1
        
        total_time = time.perf_counter() - start_time
        
        if latencies:
            return {
                "protocol": "gRPC",
                "concurrency": concurrency,
                "total_requests": num_requests,
                "successful": len(latencies),
                "failed": errors,
                "total_time_s": total_time,
                "throughput_qps": len(latencies) / total_time,
                "latency_mean_ms": np.mean(latencies),
                "latency_median_ms": np.median(latencies),
                "latency_p50_ms": np.percentile(latencies, 50),
                "latency_p95_ms": np.percentile(latencies, 95),
                "latency_p99_ms": np.percentile(latencies, 99),
                "latency_min_ms": np.min(latencies),
                "latency_max_ms": np.max(latencies)
            }
        else:
            return {"protocol": "gRPC", "error": "All requests failed"}
    
    def run_concurrency_test(
        self,
        model_name: str,
        image_path: str,
        concurrency_levels: List[int],
        requests_per_level: int
    ):
        """Test different concurrency levels"""
        
        # Preprocess image
        input_data = preprocess_image(image_path)
        
        print("Running Concurrency Benchmark")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Requests per level: {requests_per_level}")
        print(f"Concurrency levels: {concurrency_levels}")
        print("=" * 60)
        
        http_results = []
        grpc_results = []
        
        for concurrency in concurrency_levels:
            print(f"\nTesting concurrency level: {concurrency}")
            print("-" * 40)
            
            # Test HTTP
            print("  Testing HTTP...")
            http_result = self.benchmark_http(model_name, input_data, requests_per_level, concurrency)
            if "throughput_qps" in http_result:
                http_results.append(http_result)
                print(f"    Throughput: {http_result['throughput_qps']:.1f} QPS")
                print(f"    P95 Latency: {http_result['latency_p95_ms']:.1f} ms")
            
            # Test gRPC
            print("  Testing gRPC...")
            grpc_result = self.benchmark_grpc(model_name, input_data, requests_per_level, concurrency)
            if "throughput_qps" in grpc_result:
                grpc_results.append(grpc_result)
                print(f"    Throughput: {grpc_result['throughput_qps']:.1f} QPS")
                print(f"    P95 Latency: {grpc_result['latency_p95_ms']:.1f} ms")
        
        # Store results
        self.results = {"http": http_results, "grpc": grpc_results}
        
        return self.results
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate performance report"""
        
        if not self.results:
            print("No results to report")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        
        # Print comparison table
        print("\n### Throughput Comparison (QPS)")
        print("-" * 50)
        print(f"{'Concurrency':>12} {'HTTP':>12} {'gRPC':>12} {'Improvement':>12}")
        print("-" * 50)
        
        for http_res, grpc_res in zip(self.results["http"], self.results["grpc"]):
            concurrency = http_res["concurrency"]
            http_qps = http_res["throughput_qps"]
            grpc_qps = grpc_res["throughput_qps"]
            improvement = ((grpc_qps - http_qps) / http_qps) * 100
            
            print(f"{concurrency:>12} {http_qps:>11.1f} {grpc_qps:>11.1f} {improvement:>11.1f}%")
        
        print("\n### Latency Comparison (P95, ms)")
        print("-" * 50)
        print(f"{'Concurrency':>12} {'HTTP':>12} {'gRPC':>12} {'Improvement':>12}")
        print("-" * 50)
        
        for http_res, grpc_res in zip(self.results["http"], self.results["grpc"]):
            concurrency = http_res["concurrency"]
            http_p95 = http_res["latency_p95_ms"]
            grpc_p95 = grpc_res["latency_p95_ms"]
            improvement = ((http_p95 - grpc_p95) / http_p95) * 100
            
            print(f"{concurrency:>12} {http_p95:>11.1f} {grpc_p95:>11.1f} {improvement:>11.1f}%")
        
        # Find optimal concurrency
        best_http = max(self.results["http"], key=lambda x: x["throughput_qps"])
        best_grpc = max(self.results["grpc"], key=lambda x: x["throughput_qps"])
        
        print("\n### Optimal Configuration")
        print("-" * 50)
        print(f"HTTP: Concurrency={best_http['concurrency']}, Throughput={best_http['throughput_qps']:.1f} QPS")
        print(f"gRPC: Concurrency={best_grpc['concurrency']}, Throughput={best_grpc['throughput_qps']:.1f} QPS")
        
        # Save detailed report
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {output_file}")
    
    def plot_results(self, output_file: str = "benchmark_plot.png"):
        """Generate performance plots"""
        
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        concurrency_levels = [r["concurrency"] for r in self.results["http"]]
        http_throughput = [r["throughput_qps"] for r in self.results["http"]]
        grpc_throughput = [r["throughput_qps"] for r in self.results["grpc"]]
        http_p95 = [r["latency_p95_ms"] for r in self.results["http"]]
        grpc_p95 = [r["latency_p95_ms"] for r in self.results["grpc"]]
        http_p99 = [r["latency_p99_ms"] for r in self.results["http"]]
        grpc_p99 = [r["latency_p99_ms"] for r in self.results["grpc"]]
        
        # Throughput plot
        axes[0, 0].plot(concurrency_levels, http_throughput, 'o-', label='HTTP', linewidth=2)
        axes[0, 0].plot(concurrency_levels, grpc_throughput, 's-', label='gRPC', linewidth=2)
        axes[0, 0].set_xlabel('Concurrency')
        axes[0, 0].set_ylabel('Throughput (QPS)')
        axes[0, 0].set_title('Throughput vs Concurrency')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # P95 Latency plot
        axes[0, 1].plot(concurrency_levels, http_p95, 'o-', label='HTTP', linewidth=2)
        axes[0, 1].plot(concurrency_levels, grpc_p95, 's-', label='gRPC', linewidth=2)
        axes[0, 1].set_xlabel('Concurrency')
        axes[0, 1].set_ylabel('P95 Latency (ms)')
        axes[0, 1].set_title('P95 Latency vs Concurrency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # P99 Latency plot
        axes[1, 0].plot(concurrency_levels, http_p99, 'o-', label='HTTP', linewidth=2)
        axes[1, 0].plot(concurrency_levels, grpc_p99, 's-', label='gRPC', linewidth=2)
        axes[1, 0].set_xlabel('Concurrency')
        axes[1, 0].set_ylabel('P99 Latency (ms)')
        axes[1, 0].set_title('P99 Latency vs Concurrency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Latency vs Throughput
        axes[1, 1].plot(http_throughput, http_p95, 'o-', label='HTTP', linewidth=2)
        axes[1, 1].plot(grpc_throughput, grpc_p95, 's-', label='gRPC', linewidth=2)
        axes[1, 1].set_xlabel('Throughput (QPS)')
        axes[1, 1].set_ylabel('P95 Latency (ms)')
        axes[1, 1].set_title('Latency vs Throughput Trade-off')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle('Triton Inference Server Performance Benchmark', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Performance plots saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Triton Benchmark Client')
    parser.add_argument('--http-server', type=str, default='localhost:8000',
                        help='HTTP server URL')
    parser.add_argument('--grpc-server', type=str, default='localhost:8001',
                        help='gRPC server URL')
    parser.add_argument('--model', type=str, default='resnet50_pytorch',
                        help='Model name')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--requests', type=int, default=1000,
                        help='Number of requests per concurrency level')
    parser.add_argument('--plot', action='store_true',
                        help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Define concurrency levels to test
    concurrency_levels = [1, 4, 8, 16, 32, 64]
    
    # Run benchmark
    benchmark = TritonBenchmark(args.http_server, args.grpc_server)
    benchmark.run_concurrency_test(
        args.model,
        args.image,
        concurrency_levels,
        args.requests
    )
    
    # Generate report
    benchmark.generate_report()
    
    # Generate plots if requested
    if args.plot:
        benchmark.plot_results()


if __name__ == '__main__':
    main()