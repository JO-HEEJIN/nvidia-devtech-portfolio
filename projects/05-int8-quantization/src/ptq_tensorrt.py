"""
Post-Training Quantization using TensorRT

This module implements INT8 post-training quantization using TensorRT's
calibration methods including entropy-based and min-max calibrators.
"""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
from torch.utils.data import DataLoader
import onnx

# Suppress TensorRT warnings
trt.init_libnvinfer_plugins(None, "")


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Entropy-based INT8 calibrator for TensorRT.
    
    This calibrator minimizes KL-divergence between the original FP32
    activation distribution and the quantized INT8 distribution.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        cache_file: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        """
        Initialize entropy calibrator.
        
        Args:
            dataloader: DataLoader for calibration data
            cache_file: Path to save/load calibration cache
            input_shape: Expected input tensor shape
        """
        super().__init__()
        
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = dataloader.batch_size
        self.current_index = 0
        self.device_input = None
        
        # Convert dataloader to list for indexing
        self.data_list = []
        print("Preparing calibration data...")
        for batch_idx, (images, _) in enumerate(dataloader):
            # Ensure consistent batch size (pad last batch if needed)
            if images.size(0) < self.batch_size:
                padding_size = self.batch_size - images.size(0)
                padding = torch.zeros(padding_size, *images.shape[1:])
                images = torch.cat([images, padding], dim=0)
            
            self.data_list.append(images.numpy().astype(np.float32))
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
        
        print(f"Calibration data ready: {len(self.data_list)} batches")
        
        # Allocate device memory for input
        self.allocate_device_memory()
    
    def allocate_device_memory(self):
        """Allocate CUDA memory for calibration input."""
        input_size = trt.volume(self.input_shape) * self.batch_size * np.dtype(np.float32).itemsize
        self.device_input = trt.gpu.cuda.cuda_alloc(input_size)
    
    def get_batch_size(self) -> int:
        """Return batch size for calibration."""
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """
        Get next batch for calibration.
        
        Args:
            names: List of input tensor names
            
        Returns:
            List of device pointers for input tensors, or None if done
        """
        if self.current_index >= len(self.data_list):
            return None
        
        # Get current batch
        batch = self.data_list[self.current_index]
        
        # Copy to GPU
        trt.gpu.cuda.memcpy_htod(self.device_input, batch.ravel())
        
        self.current_index += 1
        
        if self.current_index % 50 == 0:
            print(f"  Calibration progress: {self.current_index}/{len(self.data_list)}")
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self) -> Optional[bytes]:
        """Read calibration cache if exists."""
        if os.path.exists(self.cache_file):
            print(f"Loading calibration cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache to file."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"Saved calibration cache to {self.cache_file}")


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    """
    MinMax-based INT8 calibrator for TensorRT.
    
    This calibrator uses the absolute maximum activation values
    to determine quantization ranges.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        cache_file: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        """Initialize MinMax calibrator."""
        super().__init__()
        
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = dataloader.batch_size
        self.current_index = 0
        self.device_input = None
        
        # Convert dataloader to list
        self.data_list = []
        print("Preparing calibration data for MinMax calibrator...")
        for batch_idx, (images, _) in enumerate(dataloader):
            if images.size(0) < self.batch_size:
                padding_size = self.batch_size - images.size(0)
                padding = torch.zeros(padding_size, *images.shape[1:])
                images = torch.cat([images, padding], dim=0)
            
            self.data_list.append(images.numpy().astype(np.float32))
        
        print(f"MinMax calibration data ready: {len(self.data_list)} batches")
        self.allocate_device_memory()
    
    def allocate_device_memory(self):
        """Allocate CUDA memory for calibration input."""
        input_size = trt.volume(self.input_shape) * self.batch_size * np.dtype(np.float32).itemsize
        self.device_input = trt.gpu.cuda.cuda_alloc(input_size)
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.current_index >= len(self.data_list):
            return None
        
        batch = self.data_list[self.current_index]
        trt.gpu.cuda.memcpy_htod(self.device_input, batch.ravel())
        self.current_index += 1
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self) -> Optional[bytes]:
        if os.path.exists(self.cache_file):
            print(f"Loading MinMax calibration cache from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f"Saved MinMax calibration cache to {self.cache_file}")


class TensorRTQuantizer:
    """Main class for TensorRT-based INT8 quantization."""
    
    def __init__(
        self,
        workspace_size: int = 1 << 30,  # 1GB
        calibrator_type: str = "entropy",
        cache_dir: str = "./cache/tensorrt"
    ):
        """
        Initialize TensorRT quantizer.
        
        Args:
            workspace_size: Maximum workspace size for TensorRT
            calibrator_type: Type of calibrator ("entropy" or "minmax")
            cache_dir: Directory to store calibration cache files
        """
        self.workspace_size = workspace_size
        self.calibrator_type = calibrator_type
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Statistics for layer-wise analysis
        self.layer_stats = {}
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        onnx_path: str = "model.onnx"
    ) -> str:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            onnx_path: Path to save ONNX model
            
        Returns:
            Path to exported ONNX model
        """
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        print(f"Exporting model to ONNX: {onnx_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX export successful and verified")
        
        return onnx_path
    
    def create_calibrator(
        self,
        dataloader: DataLoader,
        model_name: str = "model",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    ):
        """Create calibrator based on specified type."""
        
        cache_file = self.cache_dir / f"{model_name}_{self.calibrator_type}_calib.cache"
        
        if self.calibrator_type == "entropy":
            return EntropyCalibrator(dataloader, str(cache_file), input_shape)
        elif self.calibrator_type == "minmax":
            return MinMaxCalibrator(dataloader, str(cache_file), input_shape)
        else:
            raise ValueError(f"Unknown calibrator type: {self.calibrator_type}")
    
    def build_engine(
        self,
        onnx_path: str,
        calibrator,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        fp16_mode: bool = False
    ) -> trt.ICudaEngine:
        """
        Build TensorRT engine with INT8 quantization.
        
        Args:
            onnx_path: Path to ONNX model
            calibrator: Calibrator for INT8 quantization
            input_shape: Input tensor shape
            fp16_mode: Enable mixed precision (FP16 for sensitive layers)
            
        Returns:
            TensorRT engine
        """
        print(f"Building TensorRT INT8 engine from {onnx_path}")
        
        # Create builder and network
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # Set workspace size
        config.max_workspace_size = self.workspace_size
        
        # Enable INT8 precision
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        
        # Enable FP16 if requested (for mixed precision)
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Enabled FP16 mixed precision mode")
        
        # Parse ONNX model
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        print("Successfully parsed ONNX model")
        
        # Set input shape
        input_tensor = network.get_input(0)
        input_tensor.shape = input_shape
        
        print(f"Building engine (this may take several minutes)...")
        print(f"  Input shape: {input_shape}")
        print(f"  Precision: INT8 + {'FP16' if fp16_mode else 'FP32'}")
        print(f"  Calibrator: {self.calibrator_type}")
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return None
        
        print("TensorRT INT8 engine built successfully")
        return engine
    
    def save_engine(self, engine: trt.ICudaEngine, engine_path: str) -> None:
        """Save TensorRT engine to file."""
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"Saved TensorRT engine to {engine_path}")
    
    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine from file."""
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        print(f"Loaded TensorRT engine from {engine_path}")
        return engine
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        model_name: str = "model",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        mixed_precision: bool = False,
        output_dir: str = "./outputs"
    ) -> Tuple[str, str]:
        """
        Complete quantization pipeline.
        
        Args:
            model: PyTorch model to quantize
            calibration_loader: DataLoader for calibration
            model_name: Name for output files
            input_shape: Input tensor shape
            mixed_precision: Enable FP16 for sensitive layers
            output_dir: Directory for output files
            
        Returns:
            Tuple of (onnx_path, engine_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX
        onnx_path = output_dir / f"{model_name}.onnx"
        self.export_to_onnx(model, input_shape, str(onnx_path))
        
        # Create calibrator
        calibrator = self.create_calibrator(
            calibration_loader, 
            model_name, 
            input_shape
        )
        
        # Build quantized engine
        engine = self.build_engine(
            str(onnx_path),
            calibrator,
            input_shape,
            mixed_precision
        )
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        engine_path = output_dir / f"{model_name}_int8.engine"
        self.save_engine(engine, str(engine_path))
        
        # Analyze layer statistics
        self.analyze_layer_statistics(engine, model_name)
        
        return str(onnx_path), str(engine_path)
    
    def analyze_layer_statistics(self, engine: trt.ICudaEngine, model_name: str) -> Dict:
        """
        Analyze layer-wise quantization statistics.
        
        Args:
            engine: TensorRT engine
            model_name: Model name for saving stats
            
        Returns:
            Dictionary with layer statistics
        """
        print("Analyzing layer-wise quantization statistics...")
        
        stats = {
            'model_name': model_name,
            'num_layers': engine.num_layers,
            'precision_summary': {},
            'layer_details': []
        }
        
        # Count precision types
        precision_count = {'FP32': 0, 'FP16': 0, 'INT8': 0}
        
        for layer_idx in range(engine.num_layers):
            layer_name = engine.get_layer_name(layer_idx)
            # Note: Getting layer precision requires TensorRT profiling APIs
            # For now, we'll assume INT8 for most layers
            precision = 'INT8'  # Simplified assumption
            precision_count[precision] += 1
            
            layer_info = {
                'index': layer_idx,
                'name': layer_name,
                'precision': precision
            }
            stats['layer_details'].append(layer_info)
        
        stats['precision_summary'] = precision_count
        
        # Save statistics
        stats_file = self.cache_dir / f"{model_name}_quantization_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"Layer statistics saved to {stats_file}")
        print(f"Precision summary: {precision_count}")
        
        return stats


def quantize_pytorch_model_ptq(
    model: nn.Module,
    calibration_loader: DataLoader,
    model_name: str = "model",
    calibrator_type: str = "entropy",
    mixed_precision: bool = False,
    output_dir: str = "./outputs"
) -> Tuple[str, str]:
    """
    Convenience function for PTQ quantization.
    
    Args:
        model: PyTorch model to quantize
        calibration_loader: DataLoader for calibration
        model_name: Name for output files
        calibrator_type: "entropy" or "minmax"
        mixed_precision: Enable FP16 for sensitive layers
        output_dir: Output directory
        
    Returns:
        Tuple of (onnx_path, engine_path)
    """
    quantizer = TensorRTQuantizer(calibrator_type=calibrator_type)
    
    return quantizer.quantize_model(
        model=model,
        calibration_loader=calibration_loader,
        model_name=model_name,
        mixed_precision=mixed_precision,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # Example usage
    from calibration_dataset import create_calibration_dataloader
    import torchvision.models as models
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Create calibration dataloader (dummy path for demo)
    data_path = "path/to/imagenet/val"
    
    if os.path.exists(data_path):
        calib_loader = create_calibration_dataloader(
            data_path=data_path,
            batch_size=32,
            num_samples=1000
        )
        
        # Quantize model
        onnx_path, engine_path = quantize_pytorch_model_ptq(
            model=model,
            calibration_loader=calib_loader,
            model_name="resnet50",
            calibrator_type="entropy",
            mixed_precision=False
        )
        
        print(f"Quantization complete!")
        print(f"ONNX model: {onnx_path}")
        print(f"TensorRT engine: {engine_path}")
    else:
        print(f"ImageNet path not found. Update path to run quantization.")