"""
Comprehensive Accuracy Evaluation for Quantized Models

This module provides detailed accuracy assessment including top-1/top-5 accuracy,
per-class analysis, confusion matrices, and statistical significance testing.
"""

import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from scipy import stats
from tqdm import tqdm
import pandas as pd


class ModelEvaluator:
    """Comprehensive model evaluation with detailed metrics."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        num_classes: int = 1000  # ImageNet default
    ):
        """
        Initialize model evaluator.
        
        Args:
            device: Device to run evaluation on
            num_classes: Number of classes in dataset
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Evaluation results storage
        self.results = {}
        self.detailed_predictions = {}
        
    def evaluate_accuracy(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate model accuracy with comprehensive metrics.
        
        Args:
            model: Model to evaluate
            dataloader: Evaluation data loader
            model_name: Name for storing results
            
        Returns:
            Dictionary with accuracy metrics
        """
        print(f"Evaluating {model_name} accuracy...")
        
        model.eval()
        model.to(self.device)
        
        # Metrics tracking
        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        # Per-class accuracy tracking
        class_correct = torch.zeros(self.num_classes)
        class_total = torch.zeros(self.num_classes)
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(data)
                
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                # Top-1 accuracy
                _, pred_top1 = torch.max(outputs, 1)
                correct_top1 += pred_top1.eq(targets).sum().item()
                
                # Top-5 accuracy
                _, pred_top5 = torch.topk(outputs, 5, dim=1)
                correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                # Per-class accuracy
                for i in range(targets.size(0)):
                    target_class = targets[i].item()
                    class_total[target_class] += 1
                    if pred_top1[i].eq(targets[i]):
                        class_correct[target_class] += 1
                
                # Store predictions for detailed analysis
                all_predictions.extend(pred_top1.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                total_samples += targets.size(0)
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        top1_accuracy = 100.0 * correct_top1 / total_samples
        top5_accuracy = 100.0 * correct_top5 / total_samples
        
        # Per-class accuracy
        per_class_accuracy = []
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                per_class_accuracy.append(acc.item())
            else:
                per_class_accuracy.append(0.0)
        
        mean_class_accuracy = np.mean([acc for acc in per_class_accuracy if acc > 0])
        
        # Confidence statistics
        confidence_stats = {
            'mean': np.mean(all_confidences),
            'std': np.std(all_confidences),
            'median': np.median(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences)
        }
        
        # Compile results
        results = {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'mean_class_accuracy': mean_class_accuracy,
            'total_samples': total_samples,
            'evaluation_time_sec': eval_time,
            'samples_per_sec': total_samples / eval_time,
            'confidence_stats': confidence_stats,
            'per_class_accuracy': per_class_accuracy
        }
        
        # Store detailed results
        self.results[model_name] = results
        self.detailed_predictions[model_name] = {
            'predictions': all_predictions,
            'targets': all_targets,
            'confidences': all_confidences
        }
        
        print(f"\nEvaluation Results for {model_name}:")
        print(f"  Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"  Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"  Mean Class Accuracy: {mean_class_accuracy:.2f}%")
        print(f"  Evaluation Time: {eval_time:.1f}s ({total_samples/eval_time:.1f} samples/sec)")
        print(f"  Mean Confidence: {confidence_stats['mean']:.3f} ± {confidence_stats['std']:.3f}")
        
        return results
    
    def compare_models(
        self,
        model_names: List[str],
        reference_model: str = None
    ) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_names: List of model names to compare
            reference_model: Reference model for relative comparison
            
        Returns:
            DataFrame with comparison results
        """
        if not all(name in self.results for name in model_names):
            missing = [name for name in model_names if name not in self.results]
            raise ValueError(f"Missing evaluation results for: {missing}")
        
        comparison_data = []
        
        for model_name in model_names:
            result = self.results[model_name]
            
            row = {
                'Model': model_name,
                'Top-1 Acc (%)': f"{result['top1_accuracy']:.2f}",
                'Top-5 Acc (%)': f"{result['top5_accuracy']:.2f}",
                'Mean Class Acc (%)': f"{result['mean_class_accuracy']:.2f}",
                'Samples/sec': f"{result['samples_per_sec']:.1f}",
                'Mean Confidence': f"{result['confidence_stats']['mean']:.3f}"
            }
            
            # Add relative metrics if reference model specified
            if reference_model and reference_model in self.results:
                ref_result = self.results[reference_model]
                
                top1_diff = result['top1_accuracy'] - ref_result['top1_accuracy']
                top5_diff = result['top5_accuracy'] - ref_result['top5_accuracy']
                speed_ratio = result['samples_per_sec'] / ref_result['samples_per_sec']
                
                row['Top-1 Diff'] = f"{top1_diff:+.2f}"
                row['Top-5 Diff'] = f"{top5_diff:+.2f}"
                row['Speed Ratio'] = f"{speed_ratio:.1f}x"
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print(f"\nModel Comparison:")
        print(df.to_string(index=False))
        
        return df
    
    def analyze_classification_errors(
        self,
        model_name: str,
        class_names: Optional[List[str]] = None,
        top_k_errors: int = 10
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Analyze classification errors in detail.
        
        Args:
            model_name: Name of model to analyze
            class_names: List of class names (optional)
            top_k_errors: Number of top error classes to report
            
        Returns:
            Dictionary with error analysis
        """
        if model_name not in self.detailed_predictions:
            raise ValueError(f"No detailed predictions found for {model_name}")
        
        predictions = self.detailed_predictions[model_name]['predictions']
        targets = self.detailed_predictions[model_name]['targets']
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Find most confused classes
        confusion_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Class-wise error rates
        class_errors = []
        for i in range(len(precision)):
            total = support[i]
            errors = total - (recall[i] * total)
            error_rate = errors / total if total > 0 else 0
            
            class_info = {
                'class_id': i,
                'class_name': class_names[i] if class_names else f"Class_{i}",
                'total_samples': int(total),
                'error_rate': error_rate,
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i]
            }
            class_errors.append(class_info)
        
        # Sort by error rate
        class_errors.sort(key=lambda x: x['error_rate'], reverse=True)
        
        analysis = {
            'confusion_matrix': cm,
            'top_confusions': confusion_pairs[:top_k_errors],
            'worst_classes': class_errors[:top_k_errors],
            'best_classes': class_errors[-top_k_errors:],
            'overall_precision': np.mean(precision),
            'overall_recall': np.mean(recall),
            'overall_f1': np.mean(f1)
        }
        
        print(f"\nError Analysis for {model_name}:")
        print(f"Overall Precision: {analysis['overall_precision']:.3f}")
        print(f"Overall Recall: {analysis['overall_recall']:.3f}")
        print(f"Overall F1: {analysis['overall_f1']:.3f}")
        
        print(f"\nTop {top_k_errors} Most Confused Class Pairs:")
        for i, (true_class, pred_class, count) in enumerate(analysis['top_confusions']):
            true_name = class_names[true_class] if class_names else f"Class_{true_class}"
            pred_name = class_names[pred_class] if class_names else f"Class_{pred_class}"
            print(f"  {i+1}. {true_name} → {pred_name}: {count} errors")
        
        print(f"\nWorst Performing Classes:")
        for i, class_info in enumerate(analysis['worst_classes']):
            print(f"  {i+1}. {class_info['class_name']}: {class_info['error_rate']:.1%} error rate")
        
        return analysis
    
    def statistical_significance_test(
        self,
        model1_name: str,
        model2_name: str,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Test statistical significance of accuracy difference between two models.
        
        Args:
            model1_name: First model name
            model2_name: Second model name
            confidence_level: Confidence level for test
            
        Returns:
            Dictionary with test results
        """
        if model1_name not in self.detailed_predictions or model2_name not in self.detailed_predictions:
            raise ValueError("Missing detailed predictions for one or both models")
        
        pred1 = np.array(self.detailed_predictions[model1_name]['predictions'])
        pred2 = np.array(self.detailed_predictions[model2_name]['predictions'])
        targets = np.array(self.detailed_predictions[model1_name]['targets'])
        
        # Per-sample correctness
        correct1 = (pred1 == targets).astype(int)
        correct2 = (pred2 == targets).astype(int)
        
        # McNemar's test for paired binary data
        # Contingency table: [[both_wrong, model1_right_model2_wrong],
        #                     [model1_wrong_model2_right, both_right]]
        
        both_wrong = np.sum((correct1 == 0) & (correct2 == 0))
        both_right = np.sum((correct1 == 1) & (correct2 == 1))
        model1_right_model2_wrong = np.sum((correct1 == 1) & (correct2 == 0))
        model1_wrong_model2_right = np.sum((correct1 == 0) & (correct2 == 1))
        
        # McNemar's test statistic
        if model1_right_model2_wrong + model1_wrong_model2_right > 0:
            mcnemar_stat = ((abs(model1_right_model2_wrong - model1_wrong_model2_right) - 1) ** 2) / \
                          (model1_right_model2_wrong + model1_wrong_model2_right)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        else:
            mcnemar_stat = 0.0
            p_value = 1.0
        
        # Effect size (difference in accuracy)
        acc1 = np.mean(correct1)
        acc2 = np.mean(correct2)
        effect_size = acc2 - acc1
        
        # Confidence interval for effect size
        n = len(targets)
        se = np.sqrt((acc1 * (1 - acc1) + acc2 * (1 - acc2)) / n)
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        ci_lower = effect_size - z_critical * se
        ci_upper = effect_size + z_critical * se
        
        results = {
            'model1_accuracy': acc1,
            'model2_accuracy': acc2,
            'accuracy_difference': effect_size,
            'mcnemar_statistic': mcnemar_stat,
            'p_value': p_value,
            'significant': p_value < (1 - confidence_level),
            'confidence_interval': (ci_lower, ci_upper),
            'contingency_table': {
                'both_wrong': both_wrong,
                'both_right': both_right,
                'model1_better': model1_right_model2_wrong,
                'model2_better': model1_wrong_model2_right
            }
        }
        
        print(f"\nStatistical Significance Test:")
        print(f"Model 1 ({model1_name}): {acc1:.1%} accuracy")
        print(f"Model 2 ({model2_name}): {acc2:.1%} accuracy")
        print(f"Difference: {effect_size:+.1%}")
        print(f"McNemar's test p-value: {p_value:.4f}")
        print(f"Significant at {confidence_level:.0%} level: {'Yes' if results['significant'] else 'No'}")
        print(f"{confidence_level:.0%} CI for difference: [{ci_lower:+.1%}, {ci_upper:+.1%}]")
        
        return results
    
    def plot_accuracy_comparison(
        self,
        model_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot accuracy comparison between models.
        
        Args:
            model_names: List of model names to compare
            save_path: Path to save plot
            figsize: Figure size
        """
        if not all(name in self.results for name in model_names):
            missing = [name for name in model_names if name not in self.results]
            raise ValueError(f"Missing evaluation results for: {missing}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Top-1 and Top-5 accuracy comparison
        top1_scores = [self.results[name]['top1_accuracy'] for name in model_names]
        top5_scores = [self.results[name]['top5_accuracy'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, top1_scores, width, label='Top-1', alpha=0.8)
        bars2 = ax1.bar(x + width/2, top5_scores, width, label='Top-5', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Top-1 vs Top-5 Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Confidence distribution
        for i, model_name in enumerate(model_names):
            confidences = self.detailed_predictions[model_name]['confidences']
            ax2.hist(confidences, bins=50, alpha=0.6, label=model_name, density=True)
        
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy comparison plot saved to {save_path}")
        
        plt.show()


def evaluate_quantized_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    output_dir: str = "./outputs/evaluation"
) -> Dict[str, Dict]:
    """
    Convenience function to evaluate multiple quantized models.
    
    Args:
        models: Dictionary mapping model names to model objects
        dataloader: Evaluation data loader
        output_dir: Output directory for results
        
    Returns:
        Dictionary with evaluation results for all models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    # Evaluate each model
    for model_name, model in models.items():
        results = evaluator.evaluate_accuracy(model, dataloader, model_name)
        all_results[model_name] = results
    
    # Generate comparison
    model_names = list(models.keys())
    comparison_df = evaluator.compare_models(model_names)
    
    # Save comparison to CSV
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModel comparison saved to {comparison_path}")
    
    # Generate plots
    plot_path = output_dir / "accuracy_comparison.png"
    evaluator.plot_accuracy_comparison(model_names, str(plot_path))
    
    return all_results


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    from calibration_dataset import create_calibration_dataloader
    
    # Create models for demo
    fp32_model = models.resnet18(pretrained=True)
    
    # Create dummy validation loader
    data_path = "path/to/imagenet/val"
    
    if os.path.exists(data_path):
        val_loader = create_calibration_dataloader(
            data_path=data_path,
            batch_size=32,
            num_samples=1000
        )
        
        # Evaluate models
        models_dict = {
            'ResNet18_FP32': fp32_model
        }
        
        results = evaluate_quantized_models(models_dict, val_loader)
        
        print(f"\nEvaluation complete!")
        for model_name, result in results.items():
            print(f"{model_name}: {result['top1_accuracy']:.2f}% Top-1 accuracy")
    else:
        print("ImageNet path not found. Update path to run evaluation.")