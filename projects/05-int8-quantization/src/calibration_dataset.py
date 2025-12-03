"""
Calibration Dataset Creation for INT8 Quantization

This module provides utilities for creating calibration datasets from ImageNet
validation set with random sampling and caching mechanisms.
"""

import os
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


class CalibrationDataset(Dataset):
    """Custom dataset for calibration data with caching support."""
    
    def __init__(
        self, 
        data_path: str, 
        num_samples: int = 1000,
        seed: int = 42,
        cache_path: Optional[str] = None
    ):
        """
        Initialize calibration dataset.
        
        Args:
            data_path: Path to ImageNet validation directory
            num_samples: Number of calibration samples to use
            seed: Random seed for reproducible sampling
            cache_path: Path to save/load cached sample indices
        """
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.seed = seed
        self.cache_path = cache_path
        
        # Standard ImageNet transforms for validation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load or create sample indices
        self.sample_indices = self._get_sample_indices()
        
        # Load full ImageNet validation dataset
        self.full_dataset = datasets.ImageFolder(
            root=str(self.data_path), 
            transform=self.transform
        )
        
    def _get_sample_indices(self) -> List[int]:
        """Get calibration sample indices with caching."""
        
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Loading cached sample indices from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if (cached_data['num_samples'] == self.num_samples and 
                    cached_data['seed'] == self.seed):
                    return cached_data['indices']
                else:
                    print("Cache parameters don't match, regenerating indices...")
        
        # Generate new sample indices
        print(f"Generating {self.num_samples} calibration sample indices...")
        
        # Get total number of samples in dataset
        temp_dataset = datasets.ImageFolder(root=str(self.data_path))
        total_samples = len(temp_dataset)
        
        # Random sampling with fixed seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Ensure we don't sample more than available
        num_samples = min(self.num_samples, total_samples)
        sample_indices = random.sample(range(total_samples), num_samples)
        
        # Cache the indices if cache path provided
        if self.cache_path:
            cache_dir = os.path.dirname(self.cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_data = {
                'indices': sample_indices,
                'num_samples': self.num_samples,
                'seed': self.seed,
                'data_path': str(self.data_path)
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached sample indices to {self.cache_path}")
        
        return sample_indices
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get calibration sample by index."""
        actual_idx = self.sample_indices[idx]
        return self.full_dataset[actual_idx]
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes in calibration set."""
        class_counts = {}
        for idx in self.sample_indices:
            _, label = self.full_dataset[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


def create_calibration_dataloader(
    data_path: str,
    batch_size: int = 32,
    num_samples: int = 1000,
    num_workers: int = 4,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """
    Create calibration DataLoader for quantization.
    
    Args:
        data_path: Path to ImageNet validation directory
        batch_size: Batch size for DataLoader
        num_samples: Number of calibration samples
        num_workers: Number of worker processes for data loading
        seed: Random seed for sampling
        cache_dir: Directory to cache sample indices
        
    Returns:
        DataLoader for calibration data
    """
    
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(
            cache_dir, 
            f"calibration_cache_{num_samples}_{seed}.pkl"
        )
    
    dataset = CalibrationDataset(
        data_path=data_path,
        num_samples=num_samples,
        seed=seed,
        cache_path=cache_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Order doesn't matter for calibration
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Created calibration DataLoader:")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {len(dataloader)}")
    
    # Print class distribution
    class_dist = dataset.get_class_distribution()
    num_classes = len(class_dist)
    print(f"  - Classes represented: {num_classes}")
    print(f"  - Samples per class (avg): {len(dataset) / num_classes:.1f}")
    
    return dataloader


def validate_calibration_data(dataloader: DataLoader) -> dict:
    """
    Validate calibration data quality.
    
    Args:
        dataloader: Calibration DataLoader
        
    Returns:
        Dictionary with validation statistics
    """
    
    print("Validating calibration data...")
    
    total_samples = 0
    mean_accumulator = torch.zeros(3)
    std_accumulator = torch.zeros(3)
    min_values = torch.full((3,), float('inf'))
    max_values = torch.full((3,), float('-inf'))
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        total_samples += batch_size
        
        # Calculate per-channel statistics
        images_flat = images.view(batch_size, 3, -1)
        
        # Update running statistics
        batch_mean = images_flat.mean(dim=[0, 2])
        batch_std = images_flat.std(dim=[0, 2])
        batch_min = images_flat.min(dim=2)[0].min(dim=0)[0]
        batch_max = images_flat.max(dim=2)[0].max(dim=0)[0]
        
        mean_accumulator += batch_mean * batch_size
        std_accumulator += batch_std * batch_size
        min_values = torch.min(min_values, batch_min)
        max_values = torch.max(max_values, batch_max)
        
        if batch_idx == 0:
            print(f"  Sample batch shape: {images.shape}")
            print(f"  Sample label range: {labels.min().item()} - {labels.max().item()}")
    
    # Final statistics
    final_mean = mean_accumulator / total_samples
    final_std = std_accumulator / total_samples
    
    stats = {
        'total_samples': total_samples,
        'mean_rgb': final_mean.tolist(),
        'std_rgb': final_std.tolist(),
        'min_values': min_values.tolist(),
        'max_values': max_values.tolist(),
        'expected_mean': [0.485, 0.456, 0.406],  # ImageNet normalization
        'expected_std': [0.229, 0.224, 0.225]
    }
    
    print("Calibration data validation complete:")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Mean RGB: {[f'{x:.3f}' for x in stats['mean_rgb']]}")
    print(f"  - Std RGB: {[f'{x:.3f}' for x in stats['std_rgb']]}")
    print(f"  - Value range: [{min_values.min():.3f}, {max_values.max():.3f}]")
    
    return stats


if __name__ == "__main__":
    # Example usage
    data_path = "path/to/imagenet/val"
    
    if os.path.exists(data_path):
        # Create calibration dataset
        calib_loader = create_calibration_dataloader(
            data_path=data_path,
            batch_size=32,
            num_samples=1000,
            cache_dir="./cache"
        )
        
        # Validate data quality
        stats = validate_calibration_data(calib_loader)
        
        print("\nCalibration dataset ready for quantization!")
    else:
        print(f"ImageNet path {data_path} not found.")
        print("Update the path or download ImageNet validation set.")