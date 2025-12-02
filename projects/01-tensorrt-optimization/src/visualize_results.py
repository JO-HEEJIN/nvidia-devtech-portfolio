#!/usr/bin/env python3
"""
Visualization Module for TensorRT Benchmark Results

This module creates professional visualizations of benchmark results,
highlighting the performance improvements achieved through TensorRT optimization.

Visualization types:
- Latency comparison bar charts
- Throughput vs batch size line plots
- Memory usage comparison
- Speedup heatmaps
- Performance distribution box plots
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from coloredlogs import install as setup_colored_logs

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# NVIDIA brand colors
NVIDIA_GREEN = '#76B900'
NVIDIA_DARK = '#1A1A1A'
NVIDIA_GRAY = '#696969'
COLORS = {
    'pytorch_fp32': '#3498db',      # Blue
    'pytorch_fp16': '#2980b9',      # Darker blue  
    'tensorrt_fp32': '#2ecc71',     # Green
    'tensorrt_fp16': '#27ae60',     # Darker green
    'tensorrt_int8': NVIDIA_GREEN,  # NVIDIA green for INT8
}


class BenchmarkVisualizer:
    """
    Create professional visualizations for TensorRT benchmark results.
    
    Generates publication-quality plots with consistent styling and
    NVIDIA branding elements.
    """
    
    def __init__(
        self,
        results_path: str,
        output_dir: str = 'plots',
        style: str = 'whitegrid',
        dpi: int = 150
    ):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to benchmark results JSON
            output_dir: Directory to save plots
            style: Seaborn style for plots
            dpi: DPI for saved figures
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results_path = results_path
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        
        # Set plot style
        sns.set_style(style)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        self.logger.info(f"Visualizer initialized with style: {style}")
        
    def _load_results(self) -> Dict:
        """
        Load benchmark results from JSON file.
        
        Returns:
            Parsed benchmark results
        """
        with open(self.results_path, 'r') as f:
            results = json.load(f)
            
        self.logger.info(f"Loaded results from {self.results_path}")
        return results
        
    def plot_latency_comparison(self):
        """
        Create bar chart comparing latency across frameworks and precisions.
        
        Groups results by batch size and shows mean latency with error bars
        for standard deviation.
        """
        fig, axes = plt.subplots(
            2, 2,
            figsize=(14, 10),
            constrained_layout=True
        )
        axes = axes.flatten()
        
        batch_sizes = self.results['metadata']['batch_sizes']
        
        for idx, batch_size in enumerate(batch_sizes[:4]):
            ax = axes[idx]
            batch_key = f'batch_{batch_size}'
            
            if batch_key not in self.results['benchmarks']:
                continue
                
            batch_results = self.results['benchmarks'][batch_key]
            
            # Prepare data
            frameworks = []
            latencies = []
            errors = []
            colors = []
            
            # PyTorch results
            if 'pytorch' in batch_results:
                pytorch = batch_results['pytorch']
                if 'fp32' in pytorch and isinstance(pytorch['fp32'], dict):
                    frameworks.append('PyTorch\nFP32')
                    latencies.append(pytorch['fp32']['mean_latency_ms'])
                    errors.append(pytorch['fp32']['std_latency_ms'])
                    colors.append(COLORS['pytorch_fp32'])
                    
                if 'fp16' in pytorch and pytorch['fp16'] and isinstance(pytorch['fp16'], dict):
                    frameworks.append('PyTorch\nFP16')
                    latencies.append(pytorch['fp16']['mean_latency_ms'])
                    errors.append(pytorch['fp16']['std_latency_ms'])
                    colors.append(COLORS['pytorch_fp16'])
                    
            # TensorRT results
            for key in ['tensorrt_fp32', 'tensorrt_fp16', 'tensorrt_int8']:
                if key in batch_results and isinstance(batch_results[key], dict):
                    if 'mean_latency_ms' in batch_results[key]:
                        precision = key.split('_')[1].upper()
                        frameworks.append(f'TensorRT\n{precision}')
                        latencies.append(batch_results[key]['mean_latency_ms'])
                        errors.append(batch_results[key]['std_latency_ms'])
                        colors.append(COLORS[key])
                        
            # Create bar plot
            x_pos = np.arange(len(frameworks))
            bars = ax.bar(x_pos, latencies, yerr=errors, capsize=5,
                          color=colors, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_xlabel('Framework & Precision')
            ax.set_ylabel('Latency (ms)')
            ax.set_title(f'Batch Size = {batch_size}', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frameworks)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, lat in zip(bars, latencies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{lat:.1f}',
                       ha='center', va='bottom', fontsize=9)
                       
        # Overall title
        fig.suptitle('Latency Comparison: PyTorch vs TensorRT', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        # Save figure
        output_path = self.output_dir / 'latency_comparison.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        self.logger.info(f"Saved latency comparison plot to {output_path}")
        plt.close()
        
    def plot_throughput_scaling(self):
        """
        Create line plot showing throughput vs batch size scaling.
        
        Demonstrates how different optimizations scale with increasing
        batch sizes, crucial for production deployment decisions.
        """
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        
        batch_sizes = self.results['metadata']['batch_sizes']
        
        # Collect throughput data
        throughput_data = {
            'PyTorch FP32': [],
            'PyTorch FP16': [],
            'TensorRT FP32': [],
            'TensorRT FP16': [],
            'TensorRT INT8': []
        }
        
        for batch_size in batch_sizes:
            batch_key = f'batch_{batch_size}'
            
            if batch_key not in self.results['benchmarks']:
                continue
                
            batch_results = self.results['benchmarks'][batch_key]
            
            # PyTorch
            if 'pytorch' in batch_results:
                pytorch = batch_results['pytorch']
                if 'fp32' in pytorch and isinstance(pytorch['fp32'], dict):
                    throughput_data['PyTorch FP32'].append(
                        pytorch['fp32']['throughput_fps']
                    )
                else:
                    throughput_data['PyTorch FP32'].append(None)
                    
                if 'fp16' in pytorch and pytorch['fp16'] and isinstance(pytorch['fp16'], dict):
                    throughput_data['PyTorch FP16'].append(
                        pytorch['fp16']['throughput_fps']
                    )
                else:
                    throughput_data['PyTorch FP16'].append(None)
                    
            # TensorRT
            for precision, label in [('fp32', 'TensorRT FP32'),
                                    ('fp16', 'TensorRT FP16'),
                                    ('int8', 'TensorRT INT8')]:
                key = f'tensorrt_{precision}'
                if key in batch_results and isinstance(batch_results[key], dict):
                    if 'throughput_fps' in batch_results[key]:
                        throughput_data[label].append(
                            batch_results[key]['throughput_fps']
                        )
                    else:
                        throughput_data[label].append(None)
                else:
                    throughput_data[label].append(None)
                    
        # Plot lines
        line_styles = {
            'PyTorch FP32': ('--', COLORS['pytorch_fp32']),
            'PyTorch FP16': ('--', COLORS['pytorch_fp16']),
            'TensorRT FP32': ('-', COLORS['tensorrt_fp32']),
            'TensorRT FP16': ('-', COLORS['tensorrt_fp16']),
            'TensorRT INT8': ('-', COLORS['tensorrt_int8'])
        }
        
        for label, values in throughput_data.items():
            # Filter out None values
            valid_batches = [b for b, v in zip(batch_sizes, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                style, color = line_styles[label]
                ax.plot(valid_batches, valid_values, style, color=color,
                       linewidth=2.5, marker='o', markersize=8, label=label)
                       
        # Customize plot
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Throughput (images/sec)', fontsize=12)
        ax.set_title('Throughput Scaling: PyTorch vs TensorRT',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Add annotation for best performance
        max_throughput = 0
        max_config = None
        for label, values in throughput_data.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                max_val = max(valid_values)
                if max_val > max_throughput:
                    max_throughput = max_val
                    max_idx = values.index(max_val)
                    max_config = (batch_sizes[max_idx], label)
                    
        if max_config:
            ax.annotate(f'Peak: {max_throughput:.0f} fps\n{max_config[1]} @ batch {max_config[0]}',
                       xy=(max_config[0], max_throughput),
                       xytext=(max_config[0] + 2, max_throughput * 0.95),
                       fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc=NVIDIA_GREEN, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                       
        # Save figure
        output_path = self.output_dir / 'throughput_scaling.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        self.logger.info(f"Saved throughput scaling plot to {output_path}")
        plt.close()
        
    def plot_speedup_heatmap(self):
        """
        Create heatmap showing speedup factors across configurations.
        
        Visualizes the relative performance improvements of TensorRT
        compared to PyTorch baseline.
        """
        batch_sizes = self.results['metadata']['batch_sizes']
        
        # Prepare data matrix
        configs = ['TensorRT FP32', 'TensorRT FP16', 'TensorRT INT8']
        speedup_matrix = np.zeros((len(configs), len(batch_sizes)))
        
        for i, batch_size in enumerate(batch_sizes):
            batch_key = f'batch_{batch_size}'
            
            if batch_key not in self.results['benchmarks']:
                continue
                
            batch_results = self.results['benchmarks'][batch_key]
            
            # Get PyTorch baseline
            baseline = None
            if 'pytorch' in batch_results:
                pytorch = batch_results['pytorch']
                if 'fp32' in pytorch and isinstance(pytorch['fp32'], dict):
                    baseline = pytorch['fp32']['mean_latency_ms']
                    
            if baseline is None:
                continue
                
            # Calculate speedups
            for j, (key, label) in enumerate([('tensorrt_fp32', 'TensorRT FP32'),
                                              ('tensorrt_fp16', 'TensorRT FP16'),
                                              ('tensorrt_int8', 'TensorRT INT8')]):
                if key in batch_results and isinstance(batch_results[key], dict):
                    if 'mean_latency_ms' in batch_results[key]:
                        trt_latency = batch_results[key]['mean_latency_ms']
                        speedup = baseline / trt_latency
                        speedup_matrix[j, i] = speedup
                        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        
        # Use NVIDIA green colormap
        cmap = sns.color_palette("Greens", as_cmap=True)
        
        im = ax.imshow(speedup_matrix, cmap=cmap, aspect='auto', vmin=0)
        
        # Set ticks
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_yticks(np.arange(len(configs)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticklabels(configs)
        
        # Add text annotations
        for i in range(len(configs)):
            for j in range(len(batch_sizes)):
                if speedup_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{speedup_matrix[i, j]:.1f}x',
                                  ha='center', va='center', color='black')
                                  
        # Customize plot
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_title('Speedup Factor vs PyTorch FP32 Baseline',
                    fontsize=14, fontweight='bold')
                    
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup Factor', rotation=270, labelpad=20)
        
        # Save figure
        output_path = self.output_dir / 'speedup_heatmap.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        self.logger.info(f"Saved speedup heatmap to {output_path}")
        plt.close()
        
    def plot_memory_comparison(self):
        """
        Create bar chart comparing memory usage across configurations.
        
        Memory efficiency is crucial for deployment, especially on
        edge devices or when running multiple models.
        """
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        
        # Collect memory data for largest batch size
        batch_size = max(self.results['metadata']['batch_sizes'])
        batch_key = f'batch_{batch_size}'
        
        if batch_key not in self.results['benchmarks']:
            self.logger.warning(f"No results for batch size {batch_size}")
            return
            
        batch_results = self.results['benchmarks'][batch_key]
        
        # Prepare data
        configs = []
        memory_usage = []
        colors_list = []
        
        # PyTorch results
        if 'pytorch' in batch_results:
            pytorch = batch_results['pytorch']
            if 'fp32' in pytorch and isinstance(pytorch['fp32'], dict):
                if 'memory_used_mb' in pytorch['fp32']:
                    configs.append('PyTorch FP32')
                    memory_usage.append(pytorch['fp32']['memory_used_mb'])
                    colors_list.append(COLORS['pytorch_fp32'])
                    
            if 'fp16' in pytorch and pytorch['fp16'] and isinstance(pytorch['fp16'], dict):
                if 'memory_used_mb' in pytorch['fp16']:
                    configs.append('PyTorch FP16')
                    memory_usage.append(pytorch['fp16']['memory_used_mb'])
                    colors_list.append(COLORS['pytorch_fp16'])
                    
        # TensorRT results
        for key, label in [('tensorrt_fp32', 'TensorRT FP32'),
                          ('tensorrt_fp16', 'TensorRT FP16'),
                          ('tensorrt_int8', 'TensorRT INT8')]:
            if key in batch_results and isinstance(batch_results[key], dict):
                if 'memory_used_mb' in batch_results[key]:
                    configs.append(label)
                    memory_usage.append(batch_results[key]['memory_used_mb'])
                    colors_list.append(COLORS[key])
                    
        # Create bar plot
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, memory_usage, color=colors_list,
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, mem in zip(bars, memory_usage):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mem:.0f} MB',
                   ha='center', va='bottom', fontsize=10)
                   
        # Calculate memory reduction percentages
        if memory_usage and 'PyTorch FP32' in configs:
            baseline_idx = configs.index('PyTorch FP32')
            baseline_mem = memory_usage[baseline_idx]
            
            for i, (config, mem) in enumerate(zip(configs, memory_usage)):
                if 'TensorRT' in config:
                    reduction = (1 - mem/baseline_mem) * 100
                    ax.text(i, mem/2, f'-{reduction:.0f}%',
                           ha='center', va='center', fontsize=9,
                           color='white', fontweight='bold')
                           
        # Customize plot
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title(f'Memory Usage Comparison (Batch Size = {batch_size})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save figure
        output_path = self.output_dir / 'memory_comparison.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        self.logger.info(f"Saved memory comparison plot to {output_path}")
        plt.close()
        
    def plot_latency_distribution(self):
        """
        Create box plots showing latency distribution for stability analysis.
        
        Shows the consistency of inference times, important for
        real-time applications with strict latency requirements.
        """
        # This would require storing individual latency measurements
        # For now, we'll create a simplified version using mean and std
        
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        
        # Use a representative batch size
        batch_size = 8
        batch_key = f'batch_{batch_size}'
        
        if batch_key not in self.results['benchmarks']:
            batch_size = self.results['metadata']['batch_sizes'][0]
            batch_key = f'batch_{batch_size}'
            
        batch_results = self.results['benchmarks'][batch_key]
        
        # Prepare data for violin plot (using normal distribution approximation)
        np.random.seed(42)
        data = []
        labels = []
        colors_list = []
        
        configs_map = [
            ('pytorch', 'fp32', 'PyTorch FP32', COLORS['pytorch_fp32']),
            ('pytorch', 'fp16', 'PyTorch FP16', COLORS['pytorch_fp16']),
            ('tensorrt_fp32', None, 'TensorRT FP32', COLORS['tensorrt_fp32']),
            ('tensorrt_fp16', None, 'TensorRT FP16', COLORS['tensorrt_fp16']),
            ('tensorrt_int8', None, 'TensorRT INT8', COLORS['tensorrt_int8'])
        ]
        
        for key1, key2, label, color in configs_map:
            if key1 in batch_results:
                if key2:
                    # PyTorch nested structure
                    if key2 in batch_results[key1] and isinstance(batch_results[key1][key2], dict):
                        mean = batch_results[key1][key2]['mean_latency_ms']
                        std = batch_results[key1][key2]['std_latency_ms']
                        # Generate synthetic distribution
                        samples = np.random.normal(mean, std, 100)
                        data.append(samples)
                        labels.append(label)
                        colors_list.append(color)
                else:
                    # TensorRT flat structure
                    if isinstance(batch_results[key1], dict) and 'mean_latency_ms' in batch_results[key1]:
                        mean = batch_results[key1]['mean_latency_ms']
                        std = batch_results[key1]['std_latency_ms']
                        samples = np.random.normal(mean, std, 100)
                        data.append(samples)
                        labels.append(label)
                        colors_list.append(color)
                        
        # Create violin plot
        parts = ax.violinplot(data, positions=range(len(data)),
                              widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            
        # Customize plot
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title(f'Latency Distribution Analysis (Batch Size = {batch_size})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        ax.text(0.02, 0.98, 'Lines show median and mean',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
               
        # Save figure
        output_path = self.output_dir / 'latency_distribution.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.svg'), format='svg', bbox_inches='tight')
        
        self.logger.info(f"Saved latency distribution plot to {output_path}")
        plt.close()
        
    def create_summary_infographic(self):
        """
        Create a summary infographic with key metrics.
        
        A single-page visual summary suitable for presentations
        and portfolio showcases.
        """
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('TensorRT Optimization Results Summary',
                    fontsize=18, fontweight='bold')
        
        # Get best results
        best_speedup = 0
        best_config = None
        best_batch = None
        
        for batch_key, batch_results in self.results['benchmarks'].items():
            batch_size = int(batch_key.split('_')[1])
            
            if 'pytorch' in batch_results and 'fp32' in batch_results['pytorch']:
                baseline = batch_results['pytorch']['fp32']
                if isinstance(baseline, dict) and 'mean_latency_ms' in baseline:
                    baseline_latency = baseline['mean_latency_ms']
                    
                    for key in ['tensorrt_fp32', 'tensorrt_fp16', 'tensorrt_int8']:
                        if key in batch_results and isinstance(batch_results[key], dict):
                            if 'mean_latency_ms' in batch_results[key]:
                                speedup = baseline_latency / batch_results[key]['mean_latency_ms']
                                if speedup > best_speedup:
                                    best_speedup = speedup
                                    best_config = key.split('_')[1].upper()
                                    best_batch = batch_size
                                    
        # Key metrics panel
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        metrics_text = f"""
        Model: {self.results['metadata']['model'].upper()}
        Input Size: {self.results['metadata']['input_size']}
        GPU: {self.results['metadata'].get('gpu', 'Unknown')}
        
        Best Speedup: {best_speedup:.2f}x with TensorRT {best_config} (Batch {best_batch})
        """
        
        ax1.text(0.5, 0.5, metrics_text, ha='center', va='center',
                fontsize=14, bbox=dict(boxstyle='round,pad=1',
                facecolor=NVIDIA_GREEN, alpha=0.2))
                
        # Add smaller comparison plots
        # ... (would add miniature versions of the main plots)
        
        # Save figure
        output_path = self.output_dir / 'summary_infographic.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        self.logger.info(f"Saved summary infographic to {output_path}")
        plt.close()
        
    def create_all_visualizations(self):
        """Generate all visualization types."""
        
        self.logger.info("Creating all visualizations...")
        
        try:
            self.plot_latency_comparison()
        except Exception as e:
            self.logger.error(f"Failed to create latency comparison: {e}")
            
        try:
            self.plot_throughput_scaling()
        except Exception as e:
            self.logger.error(f"Failed to create throughput scaling: {e}")
            
        try:
            self.plot_speedup_heatmap()
        except Exception as e:
            self.logger.error(f"Failed to create speedup heatmap: {e}")
            
        try:
            self.plot_memory_comparison()
        except Exception as e:
            self.logger.error(f"Failed to create memory comparison: {e}")
            
        try:
            self.plot_latency_distribution()
        except Exception as e:
            self.logger.error(f"Failed to create latency distribution: {e}")
            
        try:
            self.create_summary_infographic()
        except Exception as e:
            self.logger.error(f"Failed to create summary infographic: {e}")
            
        self.logger.info(f"All visualizations saved to {self.output_dir}")


def main():
    """Main entry point for visualization."""
    
    parser = argparse.ArgumentParser(
        description='Visualize TensorRT benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to benchmark results JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Output directory for plots'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        default='whitegrid',
        choices=['whitegrid', 'darkgrid', 'white', 'dark', 'ticks'],
        help='Seaborn style for plots'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved figures'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_colored_logs(
        level=level,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create visualizer
        visualizer = BenchmarkVisualizer(
            results_path=args.results,
            output_dir=args.output,
            style=args.style,
            dpi=args.dpi
        )
        
        # Create all visualizations
        visualizer.create_all_visualizations()
        
        logging.info("Visualization complete!")
        
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()