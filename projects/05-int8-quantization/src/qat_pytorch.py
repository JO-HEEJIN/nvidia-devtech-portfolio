"""
Quantization-Aware Training (QAT) using PyTorch

This module implements quantization-aware training with PyTorch's quantization APIs,
including fine-tuning loops, learning rate scheduling, and early stopping.
"""

import os
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.quantization import (
    prepare_qat, convert, get_default_qat_qconfig,
    QConfigMapping, get_default_qconfig_mapping
)
import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import (
    QConfigMapping,
    get_default_qconfig_mapping,
    prepare_fx,
    convert_fx
)
import numpy as np
from tqdm import tqdm


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait without improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        self.best_weights = copy.deepcopy(model.state_dict())


class QuantizationAwareTrainer:
    """Main class for quantization-aware training."""
    
    def __init__(
        self,
        model: nn.Module,
        qconfig_mapping: Optional[QConfigMapping] = None,
        backend: str = 'fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM
    ):
        """
        Initialize QAT trainer.
        
        Args:
            model: Model to apply QAT to
            qconfig_mapping: Quantization configuration mapping
            backend: Quantization backend
        """
        self.original_model = model
        self.backend = backend
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Default quantization configuration
        if qconfig_mapping is None:
            self.qconfig_mapping = get_default_qconfig_mapping("qat")
        else:
            self.qconfig_mapping = qconfig_mapping
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def prepare_model_for_qat(
        self,
        example_inputs: torch.Tensor,
        train_mode: bool = True
    ) -> nn.Module:
        """
        Prepare model for quantization-aware training.
        
        Args:
            example_inputs: Example input tensor for tracing
            train_mode: Whether to prepare for training or evaluation
            
        Returns:
            Prepared model with fake quantization
        """
        print("Preparing model for quantization-aware training...")
        
        # Use FX graph mode quantization for better flexibility
        model_to_quantize = copy.deepcopy(self.original_model)
        model_to_quantize.eval()
        
        # Prepare model with fake quantization
        prepared_model = prepare_fx(
            model_to_quantize,
            qconfig_mapping=self.qconfig_mapping,
            example_inputs=example_inputs
        )
        
        if train_mode:
            prepared_model.train()
        
        print("Model prepared for QAT with fake quantization enabled")
        return prepared_model
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, float]:
        """
        Train model for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
        """
        Validate model for one epoch.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            criterion: Loss function
            device: Device to validate on
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train_quantized_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        patience: int = 5,
        save_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        Train quantization-aware model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            save_path: Path to save best model
            device: Device to train on
            
        Returns:
            Trained QAT model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Starting quantization-aware training on {device}")
        print(f"Training parameters:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Patience: {patience}")
        
        # Prepare model for QAT
        example_inputs = next(iter(train_loader))[0][:1]  # Single sample
        qat_model = self.prepare_model_for_qat(example_inputs)
        qat_model.to(device)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            qat_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(
                qat_model, train_loader, criterion, optimizer, device
            )
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(
                qat_model, val_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(qat_model.state_dict())
                
                if save_path:
                    torch.save(best_model_state, save_path)
                    print(f"  Saved best model to {save_path}")
            
            # Early stopping check
            if early_stopping(val_acc, qat_model):
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return qat_model
    
    def convert_to_quantized(
        self,
        qat_model: nn.Module,
        example_inputs: torch.Tensor
    ) -> nn.Module:
        """
        Convert QAT model to quantized model.
        
        Args:
            qat_model: Trained QAT model
            example_inputs: Example inputs for conversion
            
        Returns:
            Quantized model
        """
        print("Converting QAT model to quantized model...")
        
        qat_model.eval()
        
        # Convert fake quantization to real quantization
        quantized_model = convert_fx(qat_model)
        
        print("Model conversion complete")
        return quantized_model
    
    def compare_model_sizes(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, float]:
        """
        Compare sizes of original and quantized models.
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            
        Returns:
            Dictionary with size comparison
        """
        def get_model_size(model):
            torch.save(model.state_dict(), 'temp_model.pth')
            size = os.path.getsize('temp_model.pth') / (1024 * 1024)  # MB
            os.remove('temp_model.pth')
            return size
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        size_reduction = ((original_size - quantized_size) / original_size) * 100
        compression_ratio = original_size / quantized_size
        
        comparison = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'size_reduction_percent': size_reduction,
            'compression_ratio': compression_ratio
        }
        
        print(f"\nModel Size Comparison:")
        print(f"  Original model: {original_size:.2f} MB")
        print(f"  Quantized model: {quantized_size:.2f} MB")
        print(f"  Size reduction: {size_reduction:.1f}%")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
        
        return comparison


def train_qat_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.0001,
    save_path: str = "./qat_model.pth",
    backend: str = 'fbgemm'
) -> Tuple[nn.Module, nn.Module, Dict]:
    """
    Convenience function for QAT training.
    
    Args:
        model: Model to apply QAT to
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_path: Path to save model
        backend: Quantization backend
        
    Returns:
        Tuple of (qat_model, quantized_model, training_history)
    """
    trainer = QuantizationAwareTrainer(model, backend=backend)
    
    # Train QAT model
    qat_model = trainer.train_quantized_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_path=save_path
    )
    
    # Convert to quantized model
    example_inputs = next(iter(val_loader))[0][:1]
    quantized_model = trainer.convert_to_quantized(qat_model, example_inputs)
    
    # Compare model sizes
    size_comparison = trainer.compare_model_sizes(model, quantized_model)
    
    return qat_model, quantized_model, {
        'history': trainer.history,
        'size_comparison': size_comparison
    }


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import CIFAR10  # Using CIFAR10 for demo
    
    # Load model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Prepare data (using CIFAR10 as example)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dummy data loaders for demo
    dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    if len(dataset) > 0:
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train QAT model
        qat_model, quantized_model, results = train_qat_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            learning_rate=0.0001
        )
        
        print("\nQAT training complete!")
        print(f"Training history: {len(results['history']['train_loss'])} epochs")
        print(f"Model compression: {results['size_comparison']['compression_ratio']:.1f}x")
    else:
        print("No dataset available for demo. Download CIFAR10 or provide ImageNet data.")