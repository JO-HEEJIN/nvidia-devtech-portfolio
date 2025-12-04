"""
Batch Inference System for Healthcare VLM Deployment

This module provides efficient batch processing for medical images with optimized memory management.
Designed for high-throughput medical imaging workflows in healthcare environments.

Key Features:
- Memory-efficient batch processing for large medical datasets
- Progress tracking and resumability for long-running jobs
- Medical image preprocessing and validation
- Result aggregation and export capabilities
- GPU memory optimization for large batches
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import gc
from tqdm import tqdm
import psutil
import threading
from queue import Queue
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""
    batch_size: int = 16
    max_workers: int = 4
    prefetch_factor: int = 2
    use_gpu: bool = True
    memory_limit_gb: float = 8.0
    save_intermediate: bool = True
    output_format: str = "json"  # json, csv, hdf5
    
@dataclass
class InferenceResult:
    """Single inference result."""
    image_path: str
    text_query: str
    similarity_score: float
    inference_time_ms: float
    backend: str
    timestamp: str
    metadata: Dict[str, Any] = None

class MedicalImageBatchProcessor:
    """
    Batch processor optimized for medical imaging workflows.
    
    Handles:
    - Large medical image datasets (CT, MRI, X-ray, pathology)
    - Memory-efficient processing with automatic batch size adjustment
    - Progress tracking and checkpoint/resume functionality
    - Multi-threading for I/O operations
    """
    
    def __init__(self, 
                 model_wrapper,
                 config: BatchInferenceConfig = None):
        """
        Initialize batch processor.
        
        Args:
            model_wrapper: Model wrapper instance (PyTorch/ONNX/TensorRT)
            config: Batch inference configuration
        """
        self.model_wrapper = model_wrapper
        self.config = config or BatchInferenceConfig()
        self.results = []
        self.processed_count = 0
        self.total_count = 0
        self.start_time = None
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)
        
        # Result queue for thread-safe operations
        self.result_queue = Queue()
        
        logger.info(f"Batch processor initialized - Batch size: {self.config.batch_size}")
    
    def process_medical_dataset(self,
                               image_paths: List[str],
                               text_queries: Union[str, List[str]],
                               output_path: str,
                               checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process medical image dataset in batches.
        
        Args:
            image_paths: List of medical image file paths
            text_queries: Single query or list of queries for each image
            output_path: Path to save results
            checkpoint_path: Path to checkpoint file for resumability
            
        Returns:
            Processing summary statistics
        """
        logger.info(f"Starting batch processing of {len(image_paths)} medical images...")
        
        self.total_count = len(image_paths)
        self.start_time = time.time()
        
        # Handle single text query for all images
        if isinstance(text_queries, str):
            text_queries = [text_queries] * len(image_paths)
        
        # Load checkpoint if exists
        start_index = self._load_checkpoint(checkpoint_path)
        
        # Create batches
        batches = self._create_batches(image_paths[start_index:], text_queries[start_index:])
        
        # Process batches with progress tracking
        with tqdm(total=len(image_paths), initial=start_index, desc="Processing medical images") as pbar:
            
            for batch_idx, (image_batch, text_batch) in enumerate(batches):
                try:
                    # Check memory usage
                    if not self.memory_monitor.check_memory():
                        self._adjust_batch_size()
                        continue
                    
                    # Process batch
                    batch_results = self._process_single_batch(image_batch, text_batch)
                    
                    # Update results and progress
                    self.results.extend(batch_results)
                    self.processed_count += len(batch_results)
                    pbar.update(len(batch_results))
                    
                    # Save checkpoint periodically
                    if self.config.save_intermediate and batch_idx % 10 == 0:
                        self._save_checkpoint(checkpoint_path)
                    
                    # Force garbage collection
                    if batch_idx % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Save final results
        summary = self._save_results(output_path)
        
        # Cleanup
        self._cleanup_checkpoint(checkpoint_path)
        
        logger.info(f"Batch processing completed - Processed: {self.processed_count}/{self.total_count}")
        return summary
    
    def _create_batches(self, 
                       image_paths: List[str], 
                       text_queries: List[str]) -> Iterator[Tuple[List[str], List[str]]]:
        """Create batches from image paths and text queries."""
        for i in range(0, len(image_paths), self.config.batch_size):
            end_idx = min(i + self.config.batch_size, len(image_paths))
            yield image_paths[i:end_idx], text_queries[i:end_idx]
    
    def _process_single_batch(self, 
                             image_paths: List[str], 
                             text_queries: List[str]) -> List[InferenceResult]:
        """Process a single batch of medical images."""
        batch_results = []
        
        # Load and preprocess images in parallel
        images = self._load_images_parallel(image_paths)
        
        # Run inference for each image-text pair
        for img_path, image, text_query in zip(image_paths, images, text_queries):
            if image is not None:
                try:
                    # Measure inference time
                    start_time = time.time()
                    similarity_score = self.model_wrapper.compute_similarity(image, text_query)
                    inference_time = (time.time() - start_time) * 1000  # ms
                    
                    # Create result
                    result = InferenceResult(
                        image_path=img_path,
                        text_query=text_query,
                        similarity_score=float(similarity_score),
                        inference_time_ms=inference_time,
                        backend=self.model_wrapper.__class__.__name__,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        metadata={
                            'image_shape': list(image.shape) if hasattr(image, 'shape') else None,
                            'model_info': getattr(self.model_wrapper, 'get_model_info', lambda: {})()
                        }
                    )
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                    continue
            else:
                logger.warning(f"Failed to load image: {img_path}")
        
        return batch_results
    
    def _load_images_parallel(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Load images in parallel using thread pool."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._load_single_image, path) for path in image_paths]
            images = []
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    image = future.result()
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Image loading failed: {e}")
                    images.append(None)
        
        return images
    
    def _load_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single medical image."""
        try:
            from PIL import Image
            import cv2
            
            # Handle different medical image formats
            if image_path.lower().endswith('.dcm'):
                # DICOM handling
                try:
                    import pydicom
                    dicom_data = pydicom.dcmread(image_path)
                    image_array = dicom_data.pixel_array
                    
                    # Normalize DICOM
                    image_array = ((image_array - image_array.min()) / 
                                  (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                    
                    # Convert to RGB if grayscale
                    if len(image_array.shape) == 2:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                        
                except ImportError:
                    logger.warning(f"pydicom not available, skipping DICOM file: {image_path}")
                    return None
                    
            else:
                # Standard image formats
                image = Image.open(image_path).convert('RGB')
                image_array = np.array(image)
            
            # Resize to model input size
            if hasattr(self.model_wrapper, 'preprocess'):
                # Use model's preprocessing
                processed_image = self.model_wrapper.preprocess(image_array)
            else:
                # Default preprocessing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                processed_image = transform(image_array)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _adjust_batch_size(self) -> None:
        """Dynamically adjust batch size based on memory usage."""
        if self.config.batch_size > 1:
            self.config.batch_size = max(1, self.config.batch_size // 2)
            logger.warning(f"Memory limit reached, reducing batch size to {self.config.batch_size}")
    
    def _load_checkpoint(self, checkpoint_path: Optional[str]) -> int:
        """Load processing checkpoint."""
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    
                self.processed_count = checkpoint.get('processed_count', 0)
                self.results = [InferenceResult(**r) for r in checkpoint.get('results', [])]
                
                logger.info(f"Resuming from checkpoint - Processed: {self.processed_count}")
                return self.processed_count
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        return 0
    
    def _save_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        """Save processing checkpoint."""
        if checkpoint_path:
            checkpoint_data = {
                'processed_count': self.processed_count,
                'total_count': self.total_count,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': [asdict(r) for r in self.results]
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def _cleanup_checkpoint(self, checkpoint_path: Optional[str]) -> None:
        """Remove checkpoint file after successful completion."""
        if checkpoint_path and Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            logger.info("Checkpoint file cleaned up")
    
    def _save_results(self, output_path: str) -> Dict[str, Any]:
        """Save batch processing results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_similarity = np.mean([r.similarity_score for r in self.results]) if self.results else 0
        avg_inference_time = np.mean([r.inference_time_ms for r in self.results]) if self.results else 0
        
        summary = {
            'total_processed': len(self.results),
            'total_time_seconds': total_time,
            'average_similarity_score': avg_similarity,
            'average_inference_time_ms': avg_inference_time,
            'throughput_images_per_second': len(self.results) / total_time if total_time > 0 else 0,
            'backend': self.model_wrapper.__class__.__name__,
            'batch_size': self.config.batch_size,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save based on output format
        if self.config.output_format == "json":
            self._save_json_results(output_path, summary)
        elif self.config.output_format == "csv":
            self._save_csv_results(output_path, summary)
        elif self.config.output_format == "hdf5":
            self._save_hdf5_results(output_path, summary)
        
        logger.info(f"Results saved to {output_path}")
        return summary
    
    def _save_json_results(self, output_path: Path, summary: Dict) -> None:
        """Save results in JSON format."""
        output_data = {
            'summary': summary,
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _save_csv_results(self, output_path: Path, summary: Dict) -> None:
        """Save results in CSV format."""
        import pandas as pd
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        # Save summary separately
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_path.with_suffix('_summary.csv'), index=False)
    
    def _save_hdf5_results(self, output_path: Path, summary: Dict) -> None:
        """Save results in HDF5 format for large datasets."""
        try:
            import h5py
            
            with h5py.File(output_path.with_suffix('.h5'), 'w') as f:
                # Save summary
                summary_group = f.create_group('summary')
                for key, value in summary.items():
                    summary_group.attrs[key] = value
                
                # Save results
                results_group = f.create_group('results')
                
                # Convert results to arrays
                image_paths = [r.image_path for r in self.results]
                text_queries = [r.text_query for r in self.results]
                similarity_scores = [r.similarity_score for r in self.results]
                inference_times = [r.inference_time_ms for r in self.results]
                
                results_group.create_dataset('image_paths', data=image_paths)
                results_group.create_dataset('text_queries', data=text_queries)
                results_group.create_dataset('similarity_scores', data=similarity_scores)
                results_group.create_dataset('inference_times_ms', data=inference_times)
                
        except ImportError:
            logger.warning("h5py not available, falling back to JSON format")
            self._save_json_results(output_path, summary)


class MemoryMonitor:
    """Monitor system memory usage during batch processing."""
    
    def __init__(self, memory_limit_gb: float):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        # Check system memory
        memory_usage = psutil.virtual_memory()
        if memory_usage.available < self.memory_limit_bytes:
            return False
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            
            if gpu_memory / gpu_total > 0.9:  # 90% GPU memory usage
                return False
        
        return True
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            'cpu_memory_used_gb': psutil.virtual_memory().used / 1024**3,
            'cpu_memory_available_gb': psutil.virtual_memory().available / 1024**3,
            'cpu_memory_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_cached() / 1024**3
            })
        
        return stats


if __name__ == "__main__":
    # Test batch processing
    try:
        logger.info("Testing batch inference system...")
        
        # This would normally use real model wrapper
        class DummyWrapper:
            def compute_similarity(self, image, text):
                return np.random.random()
            
            def __class__(self):
                return "DummyWrapper"
        
        # Create config
        config = BatchInferenceConfig(
            batch_size=4,
            max_workers=2,
            output_format="json"
        )
        
        # Create processor
        processor = MedicalImageBatchProcessor(DummyWrapper(), config)
        
        # Test with dummy data
        dummy_images = ["image1.jpg", "image2.jpg", "image3.jpg"]
        dummy_query = "normal medical image"
        
        logger.info("Batch inference test setup completed")
        
    except Exception as e:
        logger.error(f"Batch inference test failed: {e}")
        logger.info("This is expected without proper model setup")