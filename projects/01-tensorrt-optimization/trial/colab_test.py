#!/usr/bin/env python3
"""
Google Colab에서 실제 GPU 테스트하기 위한 스크립트
Colab에서 이 파일을 실행하면 실제 TensorRT 성능을 측정할 수 있습니다.
"""

import subprocess
import sys

def setup_environment():
    """Colab 환경 설정"""
    print("Setting up Colab environment...")
    
    # Install required packages
    commands = [
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
        "pip install tensorrt pycuda",
        "pip install onnx onnxruntime-gpu",
        "pip install coloredlogs pynvml matplotlib seaborn tabulate tqdm",
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    
    print("\n✓ Environment setup complete!")

def run_full_test():
    """전체 파이프라인 실행"""
    
    import os
    os.chdir('/content/nvidia-devtech-portfolio/projects/01-tensorrt-optimization')
    
    print("\n" + "="*60)
    print("NVIDIA TensorRT Optimization Pipeline - Real GPU Test")
    print("="*60)
    
    # 1. Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ GPU Found: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ No GPU found!")
        return
    
    # 2. Convert PyTorch to ONNX
    print("\n1. Converting PyTorch to ONNX...")
    subprocess.run([
        "python", "src/convert_to_onnx.py",
        "--model", "resnet50",
        "--output", "models/resnet50.onnx",
        "--dynamic-batch"
    ], check=True)
    
    # 3. Generate calibration data
    print("\n2. Generating calibration data...")
    subprocess.run([
        "python", "src/calibration.py",
        "--output", "calibration_images",
        "--num-images", "100"
    ], check=True)
    
    # 4. Build TensorRT engines
    print("\n3. Building TensorRT engines...")
    
    # FP32
    print("   - Building FP32 engine...")
    subprocess.run([
        "python", "src/convert_to_tensorrt.py",
        "--onnx", "models/resnet50.onnx",
        "--output", "engines/resnet50_fp32.trt",
        "--precision", "fp32"
    ], check=True)
    
    # FP16
    print("   - Building FP16 engine...")
    subprocess.run([
        "python", "src/convert_to_tensorrt.py",
        "--onnx", "models/resnet50.onnx",
        "--output", "engines/resnet50_fp16.trt",
        "--precision", "fp16"
    ], check=True)
    
    # INT8
    print("   - Building INT8 engine...")
    subprocess.run([
        "python", "src/convert_to_tensorrt.py",
        "--onnx", "models/resnet50.onnx",
        "--output", "engines/resnet50_int8.trt",
        "--precision", "int8",
        "--calibration-data", "calibration_images"
    ], check=True)
    
    # 5. Run benchmarks
    print("\n4. Running benchmarks...")
    subprocess.run([
        "python", "src/benchmark.py",
        "--pytorch-model", "resnet50",
        "--trt-engines", "engines",
        "--batch-sizes", "1", "4", "8", "16",
        "--iterations", "100",
        "--warmup", "10",
        "--output", "results/benchmark.json"
    ], check=True)
    
    # 6. Create visualizations
    print("\n5. Creating visualizations...")
    subprocess.run([
        "python", "src/visualize_results.py",
        "--results", "results/benchmark.json",
        "--output", "plots"
    ], check=True)
    
    print("\n" + "="*60)
    print("✓ All tests completed successfully!")
    print("="*60)
    
    # Display results
    import json
    with open('results/benchmark.json', 'r') as f:
        results = json.load(f)
    
    print("\n📊 Performance Summary:")
    print("-"*40)
    
    # Find best speedup
    best_speedup = 0
    best_config = ""
    
    for batch_key, batch_results in results['benchmarks'].items():
        batch_size = int(batch_key.split('_')[1])
        
        if 'pytorch' in batch_results and 'fp32' in batch_results['pytorch']:
            baseline = batch_results['pytorch']['fp32']['mean_latency_ms']
            
            for precision in ['fp32', 'fp16', 'int8']:
                key = f'tensorrt_{precision}'
                if key in batch_results and 'mean_latency_ms' in batch_results[key]:
                    speedup = baseline / batch_results[key]['mean_latency_ms']
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_config = f"TensorRT {precision.upper()} (Batch={batch_size})"
    
    print(f"🚀 Best Speedup: {best_speedup:.2f}x with {best_config}")
    
    # Show plot
    from IPython.display import Image, display
    display(Image('plots/latency_comparison.png'))
    
    return results

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    
    if IN_COLAB:
        # Check if repo already exists
        import os
        if not os.path.exists('/content/nvidia-devtech-portfolio'):
            print("Cloning repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/JO-HEEJIN/nvidia-devtech-portfolio.git",
                "/content/nvidia-devtech-portfolio"
            ], check=True)
        else:
            print("Repository already exists. Pulling latest changes...")
            subprocess.run([
                "git", "-C", "/content/nvidia-devtech-portfolio", "pull"
            ], check=True)
        
        # Setup and run
        setup_environment()
        results = run_full_test()
    else:
        print("This script is designed to run in Google Colab.")
        print("Please upload to Colab and run there for GPU access.")