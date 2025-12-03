"""
Layer-wise Sensitivity Analysis for Quantization

This module implements per-layer sensitivity analysis to identify which layers
are most sensitive to quantization and should be kept in higher precision.
"""

import os
import pickle
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class LayerSensitivityAnalyzer:
    """Analyzes quantization sensitivity on a per-layer basis."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        cache_dir: str = "./cache/sensitivity"
    ):
        """
        Initialize sensitivity analyzer.
        
        Args:
            model: Original FP32 model
            device: Device to run analysis on
            cache_dir: Directory to cache analysis results
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Analysis results
        self.layer_names = []
        self.layer_modules = []
        self.sensitivity_scores = {}
        self.baseline_accuracy = None
        
        # Extract quantizable layers
        self._extract_quantizable_layers()
    
    def _extract_quantizable_layers(self):
        """Extract layers that can be quantized."""
        quantizable_types = (
            nn.Conv2d, nn.Linear, nn.ConvTranspose2d,
            nn.BatchNorm2d, nn.ReLU, nn.ReLU6
        )
        
        for name, module in self.model.named_modules():
            if isinstance(module, quantizable_types):
                self.layer_names.append(name)
                self.layer_modules.append(module)
        
        print(f"Found {len(self.layer_names)} quantizable layers")
        
        # Print layer summary
        layer_count = {}
        for module in self.layer_modules:
            layer_type = type(module).__name__
            layer_count[layer_type] = layer_count.get(layer_type, 0) + 1
        
        print("Layer type distribution:")
        for layer_type, count in layer_count.items():
            print(f"  {layer_type}: {count}")
    
    def evaluate_model_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> float:
        """
        Evaluate model accuracy on validation set.
        
        Args:
            model: Model to evaluate
            dataloader: Validation data loader
            num_samples: Maximum number of samples to evaluate (None for all)
            
        Returns:
            Top-1 accuracy percentage
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if num_samples and total >= num_samples:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def simulate_layer_quantization(
        self,
        layer_name: str,
        bits: int = 8
    ) -> nn.Module:
        """
        Create model with specific layer quantized.
        
        Args:
            layer_name: Name of layer to quantize
            bits: Number of bits for quantization
            
        Returns:
            Model with specified layer quantized
        """
        # Create copy of model
        quantized_model = copy.deepcopy(self.model)
        
        # Apply fake quantization to specified layer
        for name, module in quantized_model.named_modules():
            if name == layer_name:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # Quantize weights
                    with torch.no_grad():
                        weight = module.weight.data
                        weight_min = weight.min()
                        weight_max = weight.max()
                        
                        # Calculate scale and zero point
                        scale = (weight_max - weight_min) / (2**bits - 1)
                        zero_point = torch.round(-weight_min / scale).clamp(0, 2**bits - 1)
                        
                        # Quantize and dequantize
                        weight_q = torch.round(weight / scale + zero_point).clamp(0, 2**bits - 1)
                        weight_dq = (weight_q - zero_point) * scale
                        
                        module.weight.data = weight_dq
                        
                        # Quantize bias if present
                        if module.bias is not None:
                            bias = module.bias.data
                            bias_min = bias.min()
                            bias_max = bias.max()
                            bias_scale = (bias_max - bias_min) / (2**bits - 1)
                            bias_zero_point = torch.round(-bias_min / bias_scale).clamp(0, 2**bits - 1)
                            bias_q = torch.round(bias / bias_scale + bias_zero_point).clamp(0, 2**bits - 1)
                            bias_dq = (bias_q - bias_zero_point) * bias_scale
                            module.bias.data = bias_dq
        
        return quantized_model
    
    def analyze_layer_sensitivity(
        self,
        validation_loader: DataLoader,
        num_samples: int = 1000,
        bits: int = 8,
        cache_results: bool = True,
        force_recompute: bool = False
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of each layer to quantization.
        
        Args:
            validation_loader: Validation data loader
            num_samples: Number of samples to use for evaluation
            bits: Number of bits for quantization simulation
            cache_results: Whether to cache results
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        cache_file = self.cache_dir / f"sensitivity_analysis_{bits}bit.pkl"
        
        # Check cache
        if not force_recompute and cache_file.exists():
            print(f"Loading cached sensitivity analysis from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.sensitivity_scores = cached_data['sensitivity_scores']
                self.baseline_accuracy = cached_data['baseline_accuracy']
                return self.sensitivity_scores
        
        print(f"Starting layer-wise sensitivity analysis ({bits}-bit quantization)")
        print(f"Using {num_samples} samples for evaluation")
        
        # Evaluate baseline accuracy
        print("Evaluating baseline FP32 accuracy...")
        self.baseline_accuracy = self.evaluate_model_accuracy(
            self.model, validation_loader, num_samples
        )
        print(f"Baseline accuracy: {self.baseline_accuracy:.2f}%")
        
        # Analyze each layer
        self.sensitivity_scores = {}
        
        progress_bar = tqdm(self.layer_names, desc="Analyzing layers")
        
        for layer_name in progress_bar:
            # Create model with this layer quantized
            quantized_model = self.simulate_layer_quantization(layer_name, bits)
            quantized_model.to(self.device)
            
            # Evaluate accuracy
            quantized_accuracy = self.evaluate_model_accuracy(
                quantized_model, validation_loader, num_samples
            )
            
            # Calculate sensitivity (accuracy drop)
            sensitivity = self.baseline_accuracy - quantized_accuracy
            self.sensitivity_scores[layer_name] = sensitivity
            
            progress_bar.set_postfix({
                'Layer': layer_name.split('.')[-1],
                'Sensitivity': f'{sensitivity:.3f}%'
            })
            
            # Clean up
            del quantized_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Cache results
        if cache_results:
            cache_data = {
                'sensitivity_scores': self.sensitivity_scores,
                'baseline_accuracy': self.baseline_accuracy,
                'bits': bits,
                'num_samples': num_samples
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cached sensitivity analysis to {cache_file}")
        
        return self.sensitivity_scores
    
    def identify_sensitive_layers(
        self,
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Identify most sensitive layers.
        
        Args:
            threshold: Sensitivity threshold (accuracy drop %)
            top_k: Return top K most sensitive layers
            
        Returns:
            List of (layer_name, sensitivity_score) tuples
        """
        if not self.sensitivity_scores:
            raise ValueError("No sensitivity analysis results. Run analyze_layer_sensitivity first.")
        
        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(
            self.sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply threshold filter
        sensitive_layers = [
            (name, score) for name, score in sorted_layers
            if score >= threshold
        ]
        
        # Apply top-k filter
        if top_k:
            sensitive_layers = sensitive_layers[:top_k]
        
        print(f"\nIdentified {len(sensitive_layers)} sensitive layers:")
        for i, (layer_name, sensitivity) in enumerate(sensitive_layers):
            print(f"  {i+1}. {layer_name}: {sensitivity:.3f}% accuracy drop")
        
        return sensitive_layers
    
    def plot_sensitivity_distribution(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot distribution of layer sensitivities.
        
        Args:
            save_path: Path to save plot
            figsize: Figure size
        """
        if not self.sensitivity_scores:
            raise ValueError("No sensitivity analysis results to plot.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Bar plot of sensitivities
        layer_names = list(self.sensitivity_scores.keys())
        sensitivities = list(self.sensitivity_scores.values())
        
        # Shorten layer names for display
        display_names = [name.split('.')[-1] for name in layer_names]
        
        bars = ax1.bar(range(len(display_names)), sensitivities)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Sensitivity (Accuracy Drop %)')
        ax1.set_title('Layer-wise Quantization Sensitivity')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by sensitivity level
        max_sensitivity = max(sensitivities)
        for bar, sensitivity in zip(bars, sensitivities):
            if sensitivity > 0.5:
                bar.set_color('red')
            elif sensitivity > 0.1:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # Plot 2: Histogram of sensitivities
        ax2.hist(sensitivities, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Sensitivity (Accuracy Drop %)')
        ax2.set_ylabel('Number of Layers')
        ax2.set_title('Distribution of Layer Sensitivities')
        ax2.axvline(np.mean(sensitivities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sensitivities):.3f}%')
        ax2.axvline(np.median(sensitivities), color='blue', linestyle='--',
                   label=f'Median: {np.median(sensitivities):.3f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity plot saved to {save_path}")
        
        plt.show()
    
    def generate_sensitivity_report(
        self,
        save_path: str = "sensitivity_report.txt"
    ) -> str:
        """
        Generate text report of sensitivity analysis.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report string
        """
        if not self.sensitivity_scores:
            raise ValueError("No sensitivity analysis results to report.")
        
        report = []
        report.append("=" * 60)
        report.append("LAYER-WISE SENSITIVITY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        sensitivities = list(self.sensitivity_scores.values())
        report.append(f"Baseline FP32 Accuracy: {self.baseline_accuracy:.2f}%")
        report.append(f"Total Layers Analyzed: {len(sensitivities)}")
        report.append(f"Mean Sensitivity: {np.mean(sensitivities):.3f}%")
        report.append(f"Median Sensitivity: {np.median(sensitivities):.3f}%")
        report.append(f"Max Sensitivity: {max(sensitivities):.3f}%")
        report.append(f"Min Sensitivity: {min(sensitivities):.3f}%")
        report.append("")
        
        # Sensitivity categories
        high_sensitivity = sum(1 for s in sensitivities if s > 1.0)
        medium_sensitivity = sum(1 for s in sensitivities if 0.1 < s <= 1.0)
        low_sensitivity = sum(1 for s in sensitivities if s <= 0.1)
        
        report.append("Sensitivity Categories:")
        report.append(f"  High Sensitivity (>1.0%): {high_sensitivity} layers")
        report.append(f"  Medium Sensitivity (0.1-1.0%): {medium_sensitivity} layers")
        report.append(f"  Low Sensitivity (≤0.1%): {low_sensitivity} layers")
        report.append("")
        
        # Top sensitive layers
        sorted_layers = sorted(
            self.sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        report.append("Top 10 Most Sensitive Layers:")
        for i, (layer_name, sensitivity) in enumerate(sorted_layers[:10]):
            report.append(f"  {i+1:2d}. {layer_name:40s} {sensitivity:6.3f}%")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        sensitive_layers = [name for name, score in sorted_layers if score > 0.5]
        if sensitive_layers:
            report.append("  - Keep the following layers in FP16/FP32:")
            for layer in sensitive_layers:
                report.append(f"    • {layer}")
        else:
            report.append("  - All layers can be quantized to INT8 with minimal impact")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Sensitivity report saved to {save_path}")
        return report_text


def analyze_model_sensitivity(
    model: nn.Module,
    validation_loader: DataLoader,
    num_samples: int = 1000,
    bits: int = 8,
    output_dir: str = "./outputs/sensitivity"
) -> Dict[str, float]:
    """
    Convenience function for sensitivity analysis.
    
    Args:
        model: Model to analyze
        validation_loader: Validation data loader
        num_samples: Number of samples for evaluation
        bits: Quantization bits
        output_dir: Output directory for results
        
    Returns:
        Dictionary of sensitivity scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer
    analyzer = LayerSensitivityAnalyzer(model, cache_dir=str(output_dir / "cache"))
    
    # Run analysis
    sensitivity_scores = analyzer.analyze_layer_sensitivity(
        validation_loader, num_samples, bits
    )
    
    # Identify sensitive layers
    sensitive_layers = analyzer.identify_sensitive_layers(threshold=0.5, top_k=10)
    
    # Generate visualizations and report
    plot_path = output_dir / "sensitivity_plot.png"
    analyzer.plot_sensitivity_distribution(save_path=str(plot_path))
    
    report_path = output_dir / "sensitivity_report.txt"
    analyzer.generate_sensitivity_report(save_path=str(report_path))
    
    return sensitivity_scores


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    from calibration_dataset import create_calibration_dataloader
    
    # Load model
    model = models.resnet50(pretrained=True)
    
    # Create validation loader (using calibration dataset as example)
    data_path = "path/to/imagenet/val"
    
    if os.path.exists(data_path):
        val_loader = create_calibration_dataloader(
            data_path=data_path,
            batch_size=32,
            num_samples=1000
        )
        
        # Run sensitivity analysis
        sensitivity_scores = analyze_model_sensitivity(
            model=model,
            validation_loader=val_loader,
            num_samples=500,
            bits=8
        )
        
        print(f"\nSensitivity analysis complete!")
        print(f"Found {len(sensitivity_scores)} layers")
        
        # Show most sensitive layers
        sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 5 most sensitive layers:")
        for name, score in sorted_scores[:5]:
            print(f"  {name}: {score:.3f}% accuracy drop")
    else:
        print("ImageNet path not found. Update path to run analysis.")