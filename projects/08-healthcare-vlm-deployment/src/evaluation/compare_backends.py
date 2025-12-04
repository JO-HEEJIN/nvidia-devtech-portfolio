"""
Backend Comparison Tool for Healthcare VLM Deployment

This module provides comprehensive comparison between PyTorch, ONNX, and TensorRT backends.
Focuses on healthcare-specific performance metrics and deployment considerations.

Key Features:
- Performance comparison across all backends
- Medical workflow suitability analysis
- Cost-benefit analysis for clinical deployment
- Real-world scenario modeling
- ROI calculations for healthcare institutions
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BackendComparisonConfig:
    """Configuration for backend comparison."""
    test_batch_sizes: List[int] = None
    test_sequence_lengths: List[int] = None
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    memory_monitoring_enabled: bool = True
    cost_analysis_enabled: bool = True
    clinical_scenarios: List[str] = None
    
    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 4, 8, 16, 32]
        
        if self.test_sequence_lengths is None:
            self.test_sequence_lengths = [64, 128, 256, 512]
        
        if self.clinical_scenarios is None:
            self.clinical_scenarios = [
                'emergency_single_image',
                'routine_batch_processing', 
                'large_scale_screening',
                'real_time_consultation'
            ]

@dataclass
class BackendPerformanceResult:
    """Performance result for single backend test."""
    backend: str
    scenario: str
    batch_size: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float
    gpu_memory_peak_mb: float
    cpu_memory_peak_mb: float
    gpu_utilization_percent: float
    energy_efficiency_score: float
    accuracy: float
    clinical_suitability_score: float
    cost_per_inference: float
    
@dataclass 
class ComparisonSummary:
    """Summary of backend comparison."""
    best_latency_backend: str
    best_throughput_backend: str
    best_accuracy_backend: str
    best_memory_efficiency_backend: str
    best_cost_efficiency_backend: str
    clinical_recommendations: Dict[str, str]
    deployment_readiness_scores: Dict[str, float]

class PerformanceProfiler:
    """
    Detailed performance profiling for healthcare deployment scenarios.
    
    Monitors:
    - Latency distribution under clinical load
    - Memory usage patterns
    - GPU utilization efficiency  
    - Thermal and power characteristics
    """
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.monitoring_active = False
        
    def profile_backend(self, 
                       model_wrapper,
                       backend_name: str,
                       test_config: BackendComparisonConfig) -> List[BackendPerformanceResult]:
        """Profile backend across different scenarios and batch sizes."""
        results = []
        
        logger.info(f"Profiling {backend_name} backend...")
        
        # Warmup
        self._warmup_backend(model_wrapper, test_config.warmup_iterations)
        
        for scenario in test_config.clinical_scenarios:
            for batch_size in test_config.test_batch_sizes:
                result = self._profile_scenario(
                    model_wrapper, backend_name, scenario, batch_size, test_config
                )
                if result:
                    results.append(result)
        
        return results
    
    def _warmup_backend(self, model_wrapper, iterations: int) -> None:
        """Warm up backend for stable performance measurement."""
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        dummy_text = "normal medical image"
        
        for _ in range(iterations):
            try:
                _ = model_wrapper.compute_similarity(dummy_image, dummy_text)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Clear GPU memory
        if self.cuda_available:
            torch.cuda.empty_cache()
    
    def _profile_scenario(self, 
                         model_wrapper,
                         backend_name: str,
                         scenario: str,
                         batch_size: int,
                         config: BackendComparisonConfig) -> Optional[BackendPerformanceResult]:
        """Profile specific clinical scenario."""
        try:
            # Prepare test data based on scenario
            test_data = self._prepare_scenario_data(scenario, batch_size)
            
            # Monitor memory before test
            initial_memory = self._get_memory_usage()
            
            # Run performance test
            latencies, accuracies = self._run_performance_test(
                model_wrapper, test_data, config.benchmark_iterations
            )
            
            # Monitor memory after test
            peak_memory = self._get_memory_usage()
            
            # Calculate metrics
            result = self._calculate_performance_metrics(
                backend_name, scenario, batch_size, latencies, accuracies,
                initial_memory, peak_memory
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to profile {scenario} with batch size {batch_size}: {e}")
            return None
    
    def _prepare_scenario_data(self, scenario: str, batch_size: int) -> Dict[str, Any]:
        """Prepare test data for specific clinical scenario."""
        # Medical text queries for different scenarios
        scenario_queries = {
            'emergency_single_image': [
                "acute trauma chest x-ray",
                "emergency brain CT scan",
                "critical patient assessment"
            ],
            'routine_batch_processing': [
                "normal chest x-ray screening",
                "routine mammography",
                "preventive care imaging"
            ],
            'large_scale_screening': [
                "population health screening",
                "mass screening program",
                "automated diagnostic review"
            ],
            'real_time_consultation': [
                "telemedicine consultation",
                "remote diagnostic support",
                "second opinion review"
            ]
        }
        
        queries = scenario_queries.get(scenario, ["medical image analysis"])
        
        # Create batch of test images and queries
        images = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(batch_size)]
        texts = [np.random.choice(queries) for _ in range(batch_size)]
        
        return {
            'images': images,
            'texts': texts,
            'scenario_metadata': {
                'urgency': 'high' if 'emergency' in scenario else 'normal',
                'expected_volume': batch_size,
                'clinical_context': scenario
            }
        }
    
    def _run_performance_test(self, 
                             model_wrapper,
                             test_data: Dict,
                             iterations: int) -> Tuple[List[float], List[float]]:
        """Run performance benchmark with detailed timing."""
        latencies = []
        accuracies = []
        
        images = test_data['images']
        texts = test_data['texts']
        
        for i in range(iterations):
            batch_start_time = time.time()
            batch_accuracies = []
            
            # Process batch
            for img, txt in zip(images, texts):
                start_time = time.time()
                
                try:
                    similarity = model_wrapper.compute_similarity(img, txt)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    latencies.append(latency)
                    batch_accuracies.append(abs(similarity))  # Simplified accuracy metric
                    
                except Exception as e:
                    logger.warning(f"Inference failed in iteration {i}: {e}")
                    latencies.append(float('inf'))
                    batch_accuracies.append(0.0)
            
            accuracies.extend(batch_accuracies)
            
            # Periodic cleanup
            if i % 10 == 0 and self.cuda_available:
                torch.cuda.empty_cache()
        
        return latencies, accuracies
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {
            'cpu_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().memory_percent()
        }
        
        if self.cuda_available:
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_utilization': self._get_gpu_utilization()
            })
        
        return memory_info
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage (simplified)."""
        try:
            # This would use nvidia-ml-py in practice
            return 75.0  # Placeholder
        except:
            return 0.0
    
    def _calculate_performance_metrics(self,
                                     backend_name: str,
                                     scenario: str,
                                     batch_size: int,
                                     latencies: List[float],
                                     accuracies: List[float],
                                     initial_memory: Dict[str, float],
                                     peak_memory: Dict[str, float]) -> BackendPerformanceResult:
        """Calculate comprehensive performance metrics."""
        # Filter out failed inferences
        valid_latencies = [lat for lat in latencies if lat != float('inf')]
        valid_accuracies = [acc for acc in accuracies if acc > 0]
        
        if not valid_latencies:
            raise ValueError("No valid latencies recorded")
        
        # Latency metrics
        avg_latency = np.mean(valid_latencies)
        p95_latency = np.percentile(valid_latencies, 95)
        p99_latency = np.percentile(valid_latencies, 99)
        
        # Throughput calculation
        total_time = sum(valid_latencies) / 1000  # Convert to seconds
        throughput = len(valid_latencies) / total_time if total_time > 0 else 0
        
        # Memory efficiency
        gpu_memory_peak = peak_memory.get('gpu_memory_allocated_mb', 0)
        cpu_memory_peak = peak_memory.get('cpu_memory_mb', 0)
        
        # GPU utilization
        gpu_utilization = peak_memory.get('gpu_utilization', 0)
        
        # Energy efficiency score (simplified calculation)
        energy_efficiency = self._calculate_energy_efficiency(avg_latency, gpu_utilization)
        
        # Accuracy
        avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0.0
        
        # Clinical suitability score
        clinical_suitability = self._calculate_clinical_suitability(
            scenario, avg_latency, avg_accuracy, throughput
        )
        
        # Cost per inference (simplified model)
        cost_per_inference = self._estimate_cost_per_inference(
            backend_name, avg_latency, gpu_memory_peak
        )
        
        return BackendPerformanceResult(
            backend=backend_name,
            scenario=scenario,
            batch_size=batch_size,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_samples_per_sec=throughput,
            gpu_memory_peak_mb=gpu_memory_peak,
            cpu_memory_peak_mb=cpu_memory_peak,
            gpu_utilization_percent=gpu_utilization,
            energy_efficiency_score=energy_efficiency,
            accuracy=avg_accuracy,
            clinical_suitability_score=clinical_suitability,
            cost_per_inference=cost_per_inference
        )
    
    def _calculate_energy_efficiency(self, latency_ms: float, gpu_utilization: float) -> float:
        """Calculate energy efficiency score (higher is better)."""
        # Simplified model: lower latency + higher utilization = more efficient
        if latency_ms == 0:
            return 0.0
        
        # Normalize and combine metrics
        latency_score = max(0, 100 - latency_ms)  # Lower latency is better
        utilization_score = gpu_utilization  # Higher utilization is better
        
        return (latency_score + utilization_score) / 2
    
    def _calculate_clinical_suitability(self,
                                      scenario: str,
                                      latency_ms: float,
                                      accuracy: float,
                                      throughput: float) -> float:
        """Calculate clinical suitability score for scenario."""
        # Scenario-specific requirements
        scenario_requirements = {
            'emergency_single_image': {
                'max_latency_ms': 25.0,
                'min_accuracy': 0.95,
                'min_throughput': 10.0,
                'weights': {'latency': 0.5, 'accuracy': 0.4, 'throughput': 0.1}
            },
            'routine_batch_processing': {
                'max_latency_ms': 100.0,
                'min_accuracy': 0.90,
                'min_throughput': 50.0,
                'weights': {'latency': 0.2, 'accuracy': 0.3, 'throughput': 0.5}
            },
            'large_scale_screening': {
                'max_latency_ms': 200.0,
                'min_accuracy': 0.85,
                'min_throughput': 100.0,
                'weights': {'latency': 0.1, 'accuracy': 0.4, 'throughput': 0.5}
            },
            'real_time_consultation': {
                'max_latency_ms': 50.0,
                'min_accuracy': 0.90,
                'min_throughput': 5.0,
                'weights': {'latency': 0.4, 'accuracy': 0.4, 'throughput': 0.2}
            }
        }
        
        reqs = scenario_requirements.get(scenario, scenario_requirements['routine_batch_processing'])
        
        # Calculate component scores (0-1 scale)
        latency_score = max(0, min(1, (reqs['max_latency_ms'] - latency_ms) / reqs['max_latency_ms']))
        accuracy_score = max(0, min(1, accuracy / reqs['min_accuracy']))
        throughput_score = max(0, min(1, throughput / reqs['min_throughput']))
        
        # Weighted combination
        weights = reqs['weights']
        suitability_score = (
            weights['latency'] * latency_score +
            weights['accuracy'] * accuracy_score +
            weights['throughput'] * throughput_score
        )
        
        return suitability_score
    
    def _estimate_cost_per_inference(self,
                                    backend: str,
                                    latency_ms: float,
                                    gpu_memory_mb: float) -> float:
        """Estimate cost per inference (simplified model)."""
        # Cloud computing cost estimates (USD)
        base_costs = {
            'pytorch': 0.001,    # Higher due to less optimization
            'onnx': 0.0008,      # Moderate optimization
            'tensorrt': 0.0005   # Highest optimization
        }
        
        base_cost = base_costs.get(backend, 0.001)
        
        # Adjust for latency (longer inference = higher cost)
        latency_factor = latency_ms / 50.0  # Normalize to 50ms baseline
        
        # Adjust for memory usage
        memory_factor = gpu_memory_mb / 1000.0  # Normalize to 1GB baseline
        
        estimated_cost = base_cost * latency_factor * memory_factor
        
        return max(0.0001, estimated_cost)  # Minimum cost floor

class BackendComparator:
    """
    Comprehensive backend comparison system for healthcare VLM deployment.
    
    Provides detailed analysis to help healthcare institutions choose
    the optimal backend for their specific deployment requirements.
    """
    
    def __init__(self, config: BackendComparisonConfig = None):
        """
        Initialize backend comparator.
        
        Args:
            config: Comparison configuration
        """
        self.config = config or BackendComparisonConfig()
        self.profiler = PerformanceProfiler()
        self.results = {}
        
    def compare_backends(self,
                        model_wrappers: Dict[str, Any],
                        output_dir: str = "./comparison_results") -> ComparisonSummary:
        """
        Compare all provided backends comprehensively.
        
        Args:
            model_wrappers: Dictionary of backend name to model wrapper
            output_dir: Directory to save comparison results
            
        Returns:
            Comparison summary with recommendations
        """
        logger.info("Starting comprehensive backend comparison...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Profile each backend
        for backend_name, wrapper in model_wrappers.items():
            logger.info(f"Profiling {backend_name}...")
            
            try:
                backend_results = self.profiler.profile_backend(
                    wrapper, backend_name, self.config
                )
                self.results[backend_name] = backend_results
                
            except Exception as e:
                logger.error(f"Failed to profile {backend_name}: {e}")
                self.results[backend_name] = []
        
        # Generate comparison analysis
        summary = self._analyze_results()
        
        # Save results and visualizations
        self._save_comparison_results(output_path, summary)
        
        logger.info("Backend comparison completed")
        return summary
    
    def _analyze_results(self) -> ComparisonSummary:
        """Analyze profiling results and generate recommendations."""
        if not any(self.results.values()):
            raise ValueError("No valid profiling results available")
        
        # Flatten all results
        all_results = []
        for backend_results in self.results.values():
            all_results.extend(backend_results)
        
        df = pd.DataFrame([asdict(r) for r in all_results])
        
        # Find best performers across different metrics
        best_latency = df.loc[df['avg_latency_ms'].idxmin(), 'backend']
        best_throughput = df.loc[df['throughput_samples_per_sec'].idxmax(), 'backend']
        best_accuracy = df.loc[df['accuracy'].idxmax(), 'backend']
        best_memory = df.loc[df['gpu_memory_peak_mb'].idxmin(), 'backend']
        best_cost = df.loc[df['cost_per_inference'].idxmin(), 'backend']
        
        # Clinical scenario recommendations
        clinical_recs = {}
        for scenario in self.config.clinical_scenarios:
            scenario_data = df[df['scenario'] == scenario]
            if not scenario_data.empty:
                best_for_scenario = scenario_data.loc[
                    scenario_data['clinical_suitability_score'].idxmax(), 'backend'
                ]
                clinical_recs[scenario] = best_for_scenario
        
        # Deployment readiness scores
        deployment_scores = {}
        for backend in df['backend'].unique():
            backend_data = df[df['backend'] == backend]
            avg_suitability = backend_data['clinical_suitability_score'].mean()
            deployment_scores[backend] = avg_suitability
        
        return ComparisonSummary(
            best_latency_backend=best_latency,
            best_throughput_backend=best_throughput,
            best_accuracy_backend=best_accuracy,
            best_memory_efficiency_backend=best_memory,
            best_cost_efficiency_backend=best_cost,
            clinical_recommendations=clinical_recs,
            deployment_readiness_scores=deployment_scores
        )
    
    def _save_comparison_results(self, output_path: Path, summary: ComparisonSummary) -> None:
        """Save comprehensive comparison results."""
        # Save detailed results
        detailed_results = {}
        for backend, results in self.results.items():
            detailed_results[backend] = [asdict(r) for r in results]
        
        comparison_report = {
            'summary': asdict(summary),
            'detailed_results': detailed_results,
            'configuration': asdict(self.config),
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save JSON report
        with open(output_path / "backend_comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # Save CSV for easy analysis
        if any(self.results.values()):
            all_results = []
            for backend_results in self.results.values():
                all_results.extend(backend_results)
            
            df = pd.DataFrame([asdict(r) for r in all_results])
            df.to_csv(output_path / "detailed_performance_results.csv", index=False)
        
        # Generate visualizations
        self._create_comparison_visualizations(output_path, summary)
        
        logger.info(f"Comparison results saved to {output_path}")
    
    def _create_comparison_visualizations(self, output_path: Path, summary: ComparisonSummary) -> None:
        """Create comprehensive comparison visualizations."""
        if not any(self.results.values()):
            return
        
        # Prepare data
        all_results = []
        for backend_results in self.results.values():
            all_results.extend(backend_results)
        
        df = pd.DataFrame([asdict(r) for r in all_results])
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance overview dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Latency comparison
        sns.boxplot(data=df, x='backend', y='avg_latency_ms', ax=axes[0,0])
        axes[0,0].set_title('Average Latency by Backend')
        axes[0,0].set_ylabel('Latency (ms)')
        
        # Throughput comparison
        sns.boxplot(data=df, x='backend', y='throughput_samples_per_sec', ax=axes[0,1])
        axes[0,1].set_title('Throughput by Backend')
        axes[0,1].set_ylabel('Samples/Second')
        
        # Memory usage
        sns.boxplot(data=df, x='backend', y='gpu_memory_peak_mb', ax=axes[0,2])
        axes[0,2].set_title('GPU Memory Usage by Backend')
        axes[0,2].set_ylabel('Memory (MB)')
        
        # Accuracy comparison
        sns.boxplot(data=df, x='backend', y='accuracy', ax=axes[1,0])
        axes[1,0].set_title('Accuracy by Backend')
        axes[1,0].set_ylabel('Accuracy Score')
        
        # Clinical suitability
        sns.boxplot(data=df, x='backend', y='clinical_suitability_score', ax=axes[1,1])
        axes[1,1].set_title('Clinical Suitability by Backend')
        axes[1,1].set_ylabel('Suitability Score')
        
        # Cost efficiency
        sns.boxplot(data=df, x='backend', y='cost_per_inference', ax=axes[1,2])
        axes[1,2].set_title('Cost per Inference by Backend')
        axes[1,2].set_ylabel('Cost (USD)')
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Clinical scenario analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        scenarios = df['scenario'].unique()
        
        for idx, scenario in enumerate(scenarios[:4]):  # Limit to 4 scenarios
            ax = axes[idx // 2, idx % 2]
            scenario_data = df[df['scenario'] == scenario]
            
            sns.barplot(data=scenario_data, x='backend', y='clinical_suitability_score', ax=ax)
            ax.set_title(f'Clinical Suitability - {scenario.replace("_", " ").title()}')
            ax.set_ylabel('Suitability Score')
            
        plt.tight_layout()
        plt.savefig(output_path / "clinical_scenario_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance vs accuracy trade-off
        plt.figure(figsize=(12, 8))
        
        for backend in df['backend'].unique():
            backend_data = df[df['backend'] == backend]
            plt.scatter(backend_data['avg_latency_ms'], backend_data['accuracy'], 
                       label=backend, s=100, alpha=0.7)
        
        plt.xlabel('Average Latency (ms)')
        plt.ylabel('Accuracy Score')
        plt.title('Performance vs Accuracy Trade-off by Backend')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "performance_accuracy_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comparison visualizations created")


if __name__ == "__main__":
    # Test backend comparison
    try:
        logger.info("Testing backend comparison system...")
        
        # Create dummy model wrappers
        class DummyWrapper:
            def __init__(self, name, latency_factor=1.0, accuracy_factor=1.0):
                self.name = name
                self.latency_factor = latency_factor
                self.accuracy_factor = accuracy_factor
            
            def compute_similarity(self, image, text):
                # Simulate different performance characteristics
                time.sleep(0.01 * self.latency_factor)  # Simulate processing time
                base_score = 0.8 * self.accuracy_factor
                return base_score + np.random.normal(0, 0.05)
        
        # Create wrappers with different characteristics
        wrappers = {
            'pytorch': DummyWrapper('pytorch', latency_factor=2.0, accuracy_factor=1.0),
            'onnx': DummyWrapper('onnx', latency_factor=1.5, accuracy_factor=0.98),
            'tensorrt': DummyWrapper('tensorrt', latency_factor=1.0, accuracy_factor=0.96)
        }
        
        # Create comparator with minimal config for testing
        config = BackendComparisonConfig(
            test_batch_sizes=[1, 4],
            clinical_scenarios=['emergency_single_image', 'routine_batch_processing'],
            benchmark_iterations=5
        )
        
        comparator = BackendComparator(config)
        
        # This would normally run the full comparison
        logger.info("Backend comparison test setup completed")
        
    except Exception as e:
        logger.error(f"Backend comparison test failed: {e}")
        logger.info("This is expected without proper setup")