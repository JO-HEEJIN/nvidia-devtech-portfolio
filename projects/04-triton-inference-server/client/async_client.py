#!/usr/bin/env python3
"""
Asynchronous Triton client for concurrent inference requests
Measures throughput under configurable load levels
"""

import asyncio
import aiohttp
import numpy as np
import time
import argparse
import json
from PIL import Image
from typing import List, Dict, Any


def preprocess_image(image_path):
    """Preprocess image for ResNet50 inference"""
    # Load and resize image
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
    
    # Convert to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array.astype(np.float32)


class AsyncTritonClient:
    """Asynchronous HTTP client for Triton Inference Server"""
    
    def __init__(self, server_url: str, concurrency_level: int = 10):
        self.server_url = server_url
        self.concurrency_level = concurrency_level
        self.semaphore = asyncio.Semaphore(concurrency_level)
        
    async def health_check(self, session: aiohttp.ClientSession) -> bool:
        """Check if server is healthy"""
        try:
            async with session.get(f"http://{self.server_url}/v2/health/ready") as response:
                return response.status == 200
        except:
            return False
    
    async def single_inference(
        self,
        session: aiohttp.ClientSession,
        model_name: str,
        input_data: np.ndarray,
        request_id: int
    ) -> Dict[str, Any]:
        """Send single inference request"""
        
        async with self.semaphore:
            # Prepare request payload
            payload = {
                "inputs": [{
                    "name": "input__0",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "data": input_data.tolist()
                }],
                "outputs": [{
                    "name": "output__0"
                }]
            }
            
            # Send request
            start_time = time.perf_counter()
            
            try:
                async with session.post(
                    f"http://{self.server_url}/v2/models/{model_name}/infer",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    result = await response.json()
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    return {
                        "request_id": request_id,
                        "status": "success",
                        "latency_ms": latency_ms,
                        "output_shape": result["outputs"][0]["shape"],
                        "top_prediction": np.argmax(result["outputs"][0]["data"])
                    }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e)
                }
    
    async def concurrent_inference(
        self,
        model_name: str,
        image_path: str,
        num_requests: int
    ) -> Dict[str, Any]:
        """Send concurrent inference requests"""
        
        # Preprocess image once
        input_data = preprocess_image(image_path)
        
        async with aiohttp.ClientSession() as session:
            # Check server health
            is_healthy = await self.health_check(session)
            if not is_healthy:
                print("Server is not healthy")
                return {}
            
            print(f"Sending {num_requests} concurrent requests...")
            print(f"Concurrency level: {self.concurrency_level}")
            
            # Create tasks for concurrent requests
            tasks = []
            for i in range(num_requests):
                task = self.single_inference(session, model_name, input_data, i)
                tasks.append(task)
            
            # Execute all tasks concurrently
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time
            
            # Calculate statistics
            successful = [r for r in results if r["status"] == "success"]
            failed = len(results) - len(successful)
            
            if successful:
                latencies = [r["latency_ms"] for r in successful]
                throughput = len(successful) / total_time
                
                stats = {
                    "total_requests": num_requests,
                    "successful": len(successful),
                    "failed": failed,
                    "concurrency_level": self.concurrency_level,
                    "total_time_seconds": total_time,
                    "throughput_qps": throughput,
                    "latency_stats": {
                        "mean_ms": np.mean(latencies),
                        "median_ms": np.median(latencies),
                        "p95_ms": np.percentile(latencies, 95),
                        "p99_ms": np.percentile(latencies, 99),
                        "min_ms": np.min(latencies),
                        "max_ms": np.max(latencies)
                    }
                }
                
                return stats
            else:
                return {"error": "All requests failed"}


async def measure_throughput_scaling(
    server_url: str,
    model_name: str,
    image_path: str,
    concurrency_levels: List[int],
    requests_per_level: int
):
    """Measure throughput at different concurrency levels"""
    
    print("Measuring throughput scaling...")
    print("=" * 50)
    
    results = []
    
    for concurrency in concurrency_levels:
        print(f"\nConcurrency Level: {concurrency}")
        print("-" * 30)
        
        client = AsyncTritonClient(server_url, concurrency)
        stats = await client.concurrent_inference(model_name, image_path, requests_per_level)
        
        if "throughput_qps" in stats:
            print(f"Throughput: {stats['throughput_qps']:.2f} QPS")
            print(f"Mean Latency: {stats['latency_stats']['mean_ms']:.2f} ms")
            print(f"P95 Latency: {stats['latency_stats']['p95_ms']:.2f} ms")
            print(f"P99 Latency: {stats['latency_stats']['p99_ms']:.2f} ms")
            
            results.append({
                "concurrency": concurrency,
                "throughput": stats['throughput_qps'],
                "mean_latency": stats['latency_stats']['mean_ms'],
                "p95_latency": stats['latency_stats']['p95_ms'],
                "p99_latency": stats['latency_stats']['p99_ms']
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("Throughput Scaling Summary:")
    print("=" * 50)
    print(f"{'Concurrency':>12} {'Throughput':>12} {'Mean Lat':>10} {'P95 Lat':>10} {'P99 Lat':>10}")
    print("-" * 54)
    
    for r in results:
        print(f"{r['concurrency']:>12} {r['throughput']:>11.1f} {r['mean_latency']:>9.1f} {r['p95_latency']:>9.1f} {r['p99_latency']:>9.1f}")
    
    # Find optimal concurrency
    if results:
        best = max(results, key=lambda x: x['throughput'])
        print(f"\nOptimal concurrency: {best['concurrency']} ({best['throughput']:.1f} QPS)")


async def main():
    parser = argparse.ArgumentParser(description='Async Triton Client')
    parser.add_argument('--server', type=str, default='localhost:8000',
                        help='Triton server URL')
    parser.add_argument('--model', type=str, default='resnet50_pytorch',
                        help='Model name')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--requests', type=int, default=100,
                        help='Number of requests to send')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Concurrency level')
    parser.add_argument('--scale-test', action='store_true',
                        help='Test throughput scaling')
    
    args = parser.parse_args()
    
    if args.scale_test:
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
        await measure_throughput_scaling(
            args.server,
            args.model,
            args.image,
            concurrency_levels,
            args.requests
        )
    else:
        # Single test with specified concurrency
        client = AsyncTritonClient(args.server, args.concurrency)
        stats = await client.concurrent_inference(args.model, args.image, args.requests)
        
        if "throughput_qps" in stats:
            print("\nPerformance Summary:")
            print("=" * 40)
            print(f"Total Requests: {stats['total_requests']}")
            print(f"Successful: {stats['successful']}")
            print(f"Failed: {stats['failed']}")
            print(f"Concurrency: {stats['concurrency_level']}")
            print(f"Total Time: {stats['total_time_seconds']:.2f} seconds")
            print(f"Throughput: {stats['throughput_qps']:.2f} QPS")
            print("\nLatency Statistics:")
            print(f"  Mean: {stats['latency_stats']['mean_ms']:.2f} ms")
            print(f"  Median: {stats['latency_stats']['median_ms']:.2f} ms")
            print(f"  P95: {stats['latency_stats']['p95_ms']:.2f} ms")
            print(f"  P99: {stats['latency_stats']['p99_ms']:.2f} ms")
            print(f"  Min: {stats['latency_stats']['min_ms']:.2f} ms")
            print(f"  Max: {stats['latency_stats']['max_ms']:.2f} ms")


if __name__ == '__main__':
    asyncio.run(main())