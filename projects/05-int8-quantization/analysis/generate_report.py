"""
Automated Report Generation for Quantization Analysis

This module generates comprehensive markdown reports with all quantization
results, including tables, charts, and recommendations.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from compare_methods import QuantizationResults
from accuracy_evaluation import ModelEvaluator
from sensitivity_analysis import LayerSensitivityAnalyzer
from mixed_precision import PrecisionConstraints


class QuantizationReportGenerator:
    """Generate comprehensive quantization analysis reports."""
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report data storage
        self.data = {
            'metadata': {},
            'baseline_results': {},
            'quantization_results': {},
            'sensitivity_analysis': {},
            'mixed_precision_analysis': {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        # Setup matplotlib for report figures
        plt.style.use('default')
        sns.set_palette("husl")
    
    def add_metadata(
        self,
        model_name: str,
        dataset: str = "ImageNet",
        total_samples: int = None,
        experiment_date: str = None
    ):
        """Add experiment metadata to report."""
        self.data['metadata'] = {
            'model_name': model_name,
            'dataset': dataset,
            'total_samples': total_samples,
            'experiment_date': experiment_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'framework_versions': {
                'torch': getattr(__import__('torch'), '__version__', 'unknown'),
                'tensorrt': "8.5+",  # Approximate
                'numpy': getattr(__import__('numpy'), '__version__', 'unknown')
            }
        }
    
    def add_baseline_results(self, baseline: QuantizationResults):
        """Add baseline FP32 model results."""
        self.data['baseline_results'] = {
            'top1_accuracy': baseline.top1_accuracy,
            'top5_accuracy': baseline.top5_accuracy,
            'model_size_mb': baseline.model_size_mb,
            'inference_time_ms': baseline.inference_time_ms
        }
    
    def add_quantization_results(self, results: Dict[str, QuantizationResults]):
        """Add quantization method results."""
        self.data['quantization_results'] = {}
        
        for method_name, result in results.items():
            if result:  # Skip None results
                self.data['quantization_results'][method_name] = {
                    'method_name': result.method_name,
                    'top1_accuracy': result.top1_accuracy,
                    'top5_accuracy': result.top5_accuracy,
                    'accuracy_drop': result.accuracy_drop,
                    'model_size_mb': result.model_size_mb,
                    'compression_ratio': result.compression_ratio,
                    'inference_time_ms': result.inference_time_ms,
                    'speedup_factor': result.speedup_factor,
                    'quantization_time_sec': result.quantization_time_sec
                }
    
    def add_sensitivity_analysis(self, sensitivity_scores: Dict[str, float]):
        """Add layer sensitivity analysis results."""
        # Summary statistics
        scores = list(sensitivity_scores.values())
        
        self.data['sensitivity_analysis'] = {
            'total_layers': len(scores),
            'mean_sensitivity': sum(scores) / len(scores),
            'max_sensitivity': max(scores),
            'min_sensitivity': min(scores),
            'high_sensitivity_layers': sum(1 for s in scores if s > 1.0),
            'medium_sensitivity_layers': sum(1 for s in scores if 0.1 < s <= 1.0),
            'low_sensitivity_layers': sum(1 for s in scores if s <= 0.1),
            'layer_scores': sensitivity_scores
        }
        
        # Identify top sensitive layers
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        self.data['sensitivity_analysis']['top_sensitive_layers'] = sorted_layers[:10]
    
    def add_mixed_precision_analysis(self, analysis: Dict[str, Any]):
        """Add mixed precision optimization analysis."""
        self.data['mixed_precision_analysis'] = analysis
    
    def generate_summary_table(self) -> str:
        """Generate summary comparison table in markdown format."""
        if not self.data['quantization_results']:
            return "No quantization results available."
        
        # Create comparison DataFrame
        baseline = self.data['baseline_results']
        
        data = []
        # Add baseline row
        data.append({
            'Method': 'FP32 Baseline',
            'Top-1 Acc (%)': f"{baseline['top1_accuracy']:.2f}",
            'Accuracy Drop (%)': "0.00",
            'Model Size (MB)': f"{baseline['model_size_mb']:.1f}",
            'Compression': "1.0x",
            'Inference Time (ms)': f"{baseline['inference_time_ms']:.2f}",
            'Speedup': "1.0x"
        })
        
        # Add quantization results
        for result in self.data['quantization_results'].values():
            data.append({
                'Method': result['method_name'],
                'Top-1 Acc (%)': f"{result['top1_accuracy']:.2f}",
                'Accuracy Drop (%)': f"{result['accuracy_drop']:.2f}",
                'Model Size (MB)': f"{result['model_size_mb']:.1f}",
                'Compression': f"{result['compression_ratio']:.1f}x",
                'Inference Time (ms)': f"{result['inference_time_ms']:.2f}",
                'Speedup': f"{result['speedup_factor']:.1f}x"
            })
        
        df = pd.DataFrame(data)
        return df.to_markdown(index=False)
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Analyze results to generate recommendations
        if self.data['quantization_results']:
            best_method = None
            best_score = -1
            
            for method, result in self.data['quantization_results'].items():
                # Composite score: balance accuracy and compression
                score = (100 - result['accuracy_drop']) * result['compression_ratio']
                if score > best_score:
                    best_score = score
                    best_method = result['method_name']
            
            if best_method:
                recommendations.append(f"**Best Overall Method**: {best_method} offers the best balance of accuracy preservation and model compression.")
        
        # Sensitivity-based recommendations
        if self.data['sensitivity_analysis']:
            high_sens = self.data['sensitivity_analysis']['high_sensitivity_layers']
            total_layers = self.data['sensitivity_analysis']['total_layers']
            
            if high_sens > 0:
                ratio = high_sens / total_layers
                if ratio > 0.2:
                    recommendations.append(f"**Mixed Precision Recommended**: {high_sens}/{total_layers} layers show high sensitivity. Consider keeping sensitive layers in FP16 or FP32.")
                else:
                    recommendations.append(f"**Aggressive Quantization Viable**: Only {high_sens}/{total_layers} layers show high sensitivity. INT8 quantization should work well.")
        
        # Performance-based recommendations
        baseline = self.data['baseline_results']
        if baseline.get('model_size_mb', 0) > 100:
            recommendations.append("**Large Model**: Model size > 100MB. Quantization will provide significant storage and memory benefits.")
        
        if baseline.get('inference_time_ms', 0) > 50:
            recommendations.append("**Slow Inference**: Baseline inference > 50ms. Quantization will provide substantial speedup benefits.")
        
        # Accuracy threshold recommendations
        target_accuracy = baseline.get('top1_accuracy', 0) - 1.0  # 1% drop threshold
        viable_methods = []
        
        for method, result in self.data['quantization_results'].items():
            if result['top1_accuracy'] >= target_accuracy:
                viable_methods.append(result['method_name'])
        
        if viable_methods:
            recommendations.append(f"**Accuracy Target Met**: {', '.join(viable_methods)} maintain accuracy within 1% of baseline.")
        else:
            recommendations.append("**Accuracy Concern**: All methods exceed 1% accuracy drop. Consider mixed precision or QAT.")
        
        # Deployment recommendations
        recommendations.append("**Production Deployment**: Test quantized models on target hardware to validate actual performance gains.")
        recommendations.append("**Monitoring**: Implement accuracy monitoring in production to detect potential quantization issues.")
        
        return recommendations
    
    def create_visualizations(self) -> Dict[str, str]:
        """Create and save visualization plots for the report."""
        plots = {}
        
        if not self.data['quantization_results']:
            return plots
        
        # 1. Accuracy vs Compression Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = []
        accuracy_drops = []
        compressions = []
        
        for result in self.data['quantization_results'].values():
            methods.append(result['method_name'])
            accuracy_drops.append(result['accuracy_drop'])
            compressions.append(result['compression_ratio'])
        
        scatter = ax.scatter(accuracy_drops, compressions, s=100, alpha=0.7)
        
        for i, method in enumerate(methods):
            ax.annotate(method, (accuracy_drops[i], compressions[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Accuracy Drop (%)')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Accuracy vs Compression Tradeoff')
        ax.grid(True, alpha=0.3)
        
        plot_path = self.output_dir / "accuracy_vs_compression.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['accuracy_vs_compression'] = str(plot_path.name)
        
        # 2. Performance Comparison Bar Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Compression comparison
        bars1 = ax1.bar(methods, compressions, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Model Size Compression')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, comp in zip(bars1, compressions):
            height = bar.get_height()
            ax1.annotate(f'{comp:.1f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        # Speedup comparison
        speedups = [result['speedup_factor'] for result in self.data['quantization_results'].values()]
        bars2 = ax2.bar(methods, speedups, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Inference Speed Improvement')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, speed in zip(bars2, speedups):
            height = bar.get_height()
            ax2.annotate(f'{speed:.1f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['performance_comparison'] = str(plot_path.name)
        
        # 3. Sensitivity Analysis Histogram (if available)
        if self.data['sensitivity_analysis']:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scores = list(self.data['sensitivity_analysis']['layer_scores'].values())
            ax.hist(scores, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
            ax.axvline(self.data['sensitivity_analysis']['mean_sensitivity'], 
                      color='red', linestyle='--', linewidth=2, 
                      label=f"Mean: {self.data['sensitivity_analysis']['mean_sensitivity']:.3f}")
            
            ax.set_xlabel('Sensitivity Score (Accuracy Drop %)')
            ax.set_ylabel('Number of Layers')
            ax.set_title('Layer Sensitivity Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = self.output_dir / "sensitivity_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['sensitivity_distribution'] = str(plot_path.name)
        
        return plots
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        
        # Generate components
        summary_table = self.generate_summary_table()
        recommendations = self.generate_recommendations()
        plots = self.create_visualizations()
        
        # Markdown template
        template_str = """# INT8 Quantization Analysis Report

**Generated on**: {{ metadata.experiment_date }}  
**Model**: {{ metadata.model_name }}  
**Dataset**: {{ metadata.dataset }}  
{% if metadata.total_samples %}**Samples Evaluated**: {{ "{:,}".format(metadata.total_samples) }}{% endif %}

## Executive Summary

This report presents a comprehensive analysis of INT8 quantization methods applied to {{ metadata.model_name }}. The analysis compares Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and Mixed Precision approaches.

### Key Findings

{% if baseline_results %}
- **Baseline Performance**: {{ "%.2f"|format(baseline_results.top1_accuracy) }}% Top-1 accuracy, {{ "%.1f"|format(baseline_results.model_size_mb) }}MB model size
{% endif %}
{% if quantization_results %}
- **Best Compression**: {{ best_compression.method_name }} achieved {{ "%.1f"|format(best_compression.compression_ratio) }}x compression
- **Best Accuracy**: {{ best_accuracy.method_name }} preserved accuracy within {{ "%.2f"|format(best_accuracy.accuracy_drop) }}% drop
{% endif %}
{% if sensitivity_analysis %}
- **Layer Analysis**: {{ sensitivity_analysis.high_sensitivity_layers }}/{{ sensitivity_analysis.total_layers }} layers show high quantization sensitivity
{% endif %}

## Detailed Results

### Performance Comparison

{{ summary_table }}

### Visualization

{% for plot_name, plot_file in plots.items() %}
![{{ plot_name.replace('_', ' ').title() }}]({{ plot_file }})

{% endfor %}

{% if sensitivity_analysis %}
## Sensitivity Analysis

The layer-wise sensitivity analysis identified {{ sensitivity_analysis.total_layers }} quantizable layers with the following distribution:

- **High Sensitivity (>1.0% drop)**: {{ sensitivity_analysis.high_sensitivity_layers }} layers
- **Medium Sensitivity (0.1-1.0% drop)**: {{ sensitivity_analysis.medium_sensitivity_layers }} layers  
- **Low Sensitivity (≤0.1% drop)**: {{ sensitivity_analysis.low_sensitivity_layers }} layers

### Most Sensitive Layers

{% for layer_name, sensitivity in sensitivity_analysis.top_sensitive_layers[:5] %}
{{ loop.index }}. **{{ layer_name }}**: {{ "%.3f"|format(sensitivity) }}% accuracy drop
{% endfor %}
{% endif %}

{% if mixed_precision_analysis %}
## Mixed Precision Analysis

The mixed precision optimization achieved:

- **Compression Ratio**: {{ "%.1f"|format(mixed_precision_analysis.compression_ratio) }}x
- **Estimated Accuracy Drop**: {{ "%.2f"|format(mixed_precision_analysis.estimated_accuracy_drop) }}%
- **Precision Distribution**: 
  - INT8: {{ "%.1%"|format(mixed_precision_analysis.int8_ratio) }}
  - FP16: {{ mixed_precision_analysis.fp16_layers }} layers
  - FP32: {{ mixed_precision_analysis.fp32_layers }} layers
{% endif %}

## Recommendations

{% for recommendation in recommendations %}
{{ loop.index }}. {{ recommendation }}

{% endfor %}

## Technical Details

### Quantization Methods Evaluated

{% for method_name, result in quantization_results.items() %}
#### {{ result.method_name }}

- **Accuracy**: {{ "%.2f"|format(result.top1_accuracy) }}% Top-1 ({{ "%.2f"|format(result.accuracy_drop) }}% drop)
- **Compression**: {{ "%.1f"|format(result.compression_ratio) }}x size reduction
- **Performance**: {{ "%.1f"|format(result.speedup_factor) }}x inference speedup
- **Quantization Time**: {{ "%.1f"|format(result.quantization_time_sec) }}s

{% endfor %}

### Methodology

1. **Baseline Evaluation**: Original FP32 model accuracy and performance measurement
2. **Post-Training Quantization**: TensorRT-based INT8 conversion with entropy/minmax calibration
3. **Quantization-Aware Training**: PyTorch fake quantization with fine-tuning
4. **Sensitivity Analysis**: Per-layer quantization impact assessment
5. **Mixed Precision**: Optimal precision assignment based on sensitivity scores

### Hardware and Software Environment

{% for framework, version in metadata.framework_versions.items() %}
- **{{ framework.title() }}**: {{ version }}
{% endfor %}

## Conclusion

{% if best_method %}
Based on the comprehensive analysis, **{{ best_method }}** provides the optimal balance of accuracy preservation and model optimization for {{ metadata.model_name }}. 
{% endif %}

For production deployment, we recommend:

1. Validate quantized models on target hardware
2. Implement accuracy monitoring
3. Consider mixed precision for accuracy-critical applications
4. Test with representative production data

---

*Report generated by INT8 Quantization Pipeline*
*Contact: [Project Repository](https://github.com/example/quantization-pipeline)*
"""

        # Prepare template data
        template_data = self.data.copy()
        
        # Find best methods for summary
        if self.data['quantization_results']:
            best_compression = max(self.data['quantization_results'].values(), 
                                 key=lambda x: x['compression_ratio'])
            best_accuracy = min(self.data['quantization_results'].values(),
                              key=lambda x: x['accuracy_drop'])
            
            template_data['best_compression'] = best_compression
            template_data['best_accuracy'] = best_accuracy
            template_data['best_method'] = best_accuracy['method_name']
        
        template_data['summary_table'] = summary_table
        template_data['recommendations'] = recommendations
        template_data['plots'] = plots
        
        # Render template
        template = Template(template_str)
        return template.render(**template_data)
    
    def save_report(self, filename: str = "quantization_report.md") -> str:
        """
        Generate and save the complete report.
        
        Args:
            filename: Output filename for the report
            
        Returns:
            Path to saved report
        """
        report_content = self.generate_markdown_report()
        
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save raw data as JSON for programmatic access
        data_path = self.output_dir / "report_data.json"
        with open(data_path, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_data = self._convert_for_json(self.data)
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"Quantization analysis report saved to: {report_path}")
        print(f"Raw data saved to: {data_path}")
        
        return str(report_path)
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def generate_complete_report(
    model_name: str,
    baseline_results: QuantizationResults,
    quantization_results: Dict[str, QuantizationResults],
    sensitivity_scores: Optional[Dict[str, float]] = None,
    mixed_precision_analysis: Optional[Dict[str, Any]] = None,
    dataset: str = "ImageNet",
    total_samples: int = None,
    output_dir: str = "./reports"
) -> str:
    """
    Convenience function to generate complete quantization report.
    
    Args:
        model_name: Name of the model being analyzed
        baseline_results: Baseline FP32 model results
        quantization_results: Results from different quantization methods
        sensitivity_scores: Layer sensitivity scores (optional)
        mixed_precision_analysis: Mixed precision analysis results (optional)
        dataset: Dataset name
        total_samples: Total number of samples evaluated
        output_dir: Output directory for report
        
    Returns:
        Path to generated report
    """
    generator = QuantizationReportGenerator(output_dir)
    
    # Add all data to the report
    generator.add_metadata(model_name, dataset, total_samples)
    generator.add_baseline_results(baseline_results)
    generator.add_quantization_results(quantization_results)
    
    if sensitivity_scores:
        generator.add_sensitivity_analysis(sensitivity_scores)
    
    if mixed_precision_analysis:
        generator.add_mixed_precision_analysis(mixed_precision_analysis)
    
    # Generate and save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quantization_report_{model_name}_{timestamp}.md"
    
    return generator.save_report(filename)


if __name__ == "__main__":
    # Example usage
    from compare_methods import QuantizationResults
    
    # Mock data for demonstration
    baseline = QuantizationResults(
        method_name="FP32 Baseline",
        model_size_mb=98.0,
        inference_time_ms=15.2,
        top1_accuracy=76.15,
        top5_accuracy=92.87,
        accuracy_drop=0.0,
        compression_ratio=1.0,
        speedup_factor=1.0,
        quantization_time_sec=0.0
    )
    
    ptq_entropy = QuantizationResults(
        method_name="PTQ (entropy)",
        model_size_mb=24.5,
        inference_time_ms=6.1,
        top1_accuracy=75.32,
        top5_accuracy=92.15,
        accuracy_drop=0.83,
        compression_ratio=4.0,
        speedup_factor=2.5,
        quantization_time_sec=180.0
    )
    
    qat = QuantizationResults(
        method_name="QAT",
        model_size_mb=24.5,
        inference_time_ms=6.8,
        top1_accuracy=75.89,
        top5_accuracy=92.65,
        accuracy_drop=0.26,
        compression_ratio=4.0,
        speedup_factor=2.2,
        quantization_time_sec=3600.0
    )
    
    quantization_results = {
        "ptq_entropy": ptq_entropy,
        "qat": qat
    }
    
    sensitivity_scores = {
        'conv1': 2.1,
        'layer1.0.conv1': 0.3,
        'layer1.0.conv2': 0.2,
        'layer2.0.conv1': 0.8,
        'layer4.0.conv2': 0.9,
        'fc': 1.8
    }
    
    # Generate report
    report_path = generate_complete_report(
        model_name="ResNet50",
        baseline_results=baseline,
        quantization_results=quantization_results,
        sensitivity_scores=sensitivity_scores,
        total_samples=50000
    )
    
    print(f"Demo report generated: {report_path}")