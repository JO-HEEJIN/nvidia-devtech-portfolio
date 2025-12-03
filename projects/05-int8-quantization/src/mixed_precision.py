"""
Mixed Precision Optimization for Quantization

This module implements algorithms to optimally assign precision levels (FP32, FP16, INT8)
to different layers based on sensitivity analysis and performance constraints.
"""

import os
import pickle
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensitivity_analysis import LayerSensitivityAnalyzer


class PrecisionType(Enum):
    """Enumeration of precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class PrecisionConstraints:
    """Constraints for mixed precision optimization."""
    max_accuracy_drop: float = 1.0  # Maximum acceptable accuracy drop (%)
    target_compression_ratio: float = 3.0  # Target model size compression
    memory_budget_mb: Optional[float] = None  # Memory budget in MB
    latency_budget_ms: Optional[float] = None  # Latency budget in milliseconds
    min_int8_ratio: float = 0.7  # Minimum fraction of layers in INT8


@dataclass
class LayerPrecisionAssignment:
    """Precision assignment for a single layer."""
    layer_name: str
    precision: PrecisionType
    sensitivity_score: float
    size_mb: float
    estimated_latency_ms: float


class MixedPrecisionOptimizer:
    """Optimizes precision assignment for minimal accuracy loss and maximum compression."""
    
    def __init__(
        self,
        model: nn.Module,
        sensitivity_scores: Dict[str, float],
        device: Optional[torch.device] = None
    ):
        """
        Initialize mixed precision optimizer.
        
        Args:
            model: Original FP32 model
            sensitivity_scores: Layer sensitivity scores from sensitivity analysis
            device: Device to run optimization on
        """
        self.model = model
        self.sensitivity_scores = sensitivity_scores
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Layer information
        self.layer_info = self._extract_layer_info()
        
        # Precision assignments
        self.current_assignment = {}
        self.optimal_assignment = {}
        
        # Performance estimates
        self.precision_performance = self._estimate_precision_performance()
    
    def _extract_layer_info(self) -> Dict[str, Dict]:
        """Extract layer information including parameter counts and types."""
        layer_info = {}
        
        for name, module in self.model.named_modules():
            if name in self.sensitivity_scores:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Estimate memory usage for different precisions
                fp32_size = param_count * 4 / (1024 * 1024)  # MB
                fp16_size = param_count * 2 / (1024 * 1024)  # MB
                int8_size = param_count * 1 / (1024 * 1024)  # MB
                
                layer_info[name] = {
                    'module_type': type(module).__name__,
                    'param_count': param_count,
                    'fp32_size_mb': fp32_size,
                    'fp16_size_mb': fp16_size,
                    'int8_size_mb': int8_size,
                    'sensitivity': self.sensitivity_scores[name]
                }
        
        return layer_info
    
    def _estimate_precision_performance(self) -> Dict[PrecisionType, Dict[str, float]]:
        """Estimate performance characteristics for each precision type."""
        
        # Performance estimates based on typical hardware characteristics
        performance = {
            PrecisionType.FP32: {
                'throughput_factor': 1.0,
                'memory_factor': 1.0,
                'accuracy_factor': 1.0
            },
            PrecisionType.FP16: {
                'throughput_factor': 1.5,  # ~1.5x faster
                'memory_factor': 0.5,      # 50% memory usage
                'accuracy_factor': 0.999   # Minimal accuracy loss
            },
            PrecisionType.INT8: {
                'throughput_factor': 3.0,  # ~3x faster
                'memory_factor': 0.25,     # 25% memory usage
                'accuracy_factor': 0.98    # Some accuracy loss
            }
        }
        
        return performance
    
    def greedy_assignment(
        self,
        constraints: PrecisionConstraints
    ) -> Dict[str, PrecisionType]:
        """
        Greedy algorithm for precision assignment.
        
        Assigns precision based on sensitivity scores, starting with most sensitive
        layers in higher precision.
        
        Args:
            constraints: Optimization constraints
            
        Returns:
            Dictionary mapping layer names to precision types
        """
        print("Running greedy precision assignment...")
        
        # Sort layers by sensitivity (descending)
        sorted_layers = sorted(
            self.sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        assignment = {}
        cumulative_accuracy_drop = 0.0
        total_compression_achieved = 0.0
        int8_layer_count = 0
        
        for layer_name, sensitivity in sorted_layers:
            layer_info = self.layer_info[layer_name]
            
            # Decision logic based on sensitivity and constraints
            if sensitivity > 1.0:  # Very sensitive - keep in FP32
                precision = PrecisionType.FP32
                accuracy_impact = 0.0
            elif sensitivity > 0.5:  # Moderately sensitive - use FP16
                precision = PrecisionType.FP16
                accuracy_impact = sensitivity * 0.1  # Reduced impact with FP16
            else:  # Low sensitivity - can use INT8
                precision = PrecisionType.INT8
                accuracy_impact = sensitivity
                int8_layer_count += 1
            
            # Check constraints
            potential_accuracy_drop = cumulative_accuracy_drop + accuracy_impact
            
            if potential_accuracy_drop > constraints.max_accuracy_drop:
                # Would exceed accuracy constraint, use higher precision
                if precision == PrecisionType.INT8:
                    precision = PrecisionType.FP16
                    accuracy_impact = sensitivity * 0.1
                    int8_layer_count -= 1
                elif precision == PrecisionType.FP16:
                    precision = PrecisionType.FP32
                    accuracy_impact = 0.0
            
            assignment[layer_name] = precision
            cumulative_accuracy_drop += accuracy_impact
        
        # Check minimum INT8 ratio constraint
        total_layers = len(sorted_layers)
        int8_ratio = int8_layer_count / total_layers if total_layers > 0 else 0
        
        if int8_ratio < constraints.min_int8_ratio:
            print(f"Warning: INT8 ratio {int8_ratio:.2f} below target {constraints.min_int8_ratio}")
        
        print(f"Greedy assignment complete:")
        print(f"  - Estimated accuracy drop: {cumulative_accuracy_drop:.2f}%")
        print(f"  - INT8 layer ratio: {int8_ratio:.2f}")
        
        return assignment
    
    def dynamic_programming_assignment(
        self,
        constraints: PrecisionConstraints
    ) -> Dict[str, PrecisionType]:
        """
        Dynamic programming approach for optimal precision assignment.
        
        Finds globally optimal solution considering all layer interactions.
        
        Args:
            constraints: Optimization constraints
            
        Returns:
            Optimal precision assignment
        """
        print("Running dynamic programming optimization...")
        
        layers = list(self.sensitivity_scores.keys())
        n_layers = len(layers)
        
        if n_layers > 20:  # Limit DP to reasonable size
            print(f"Too many layers ({n_layers}) for DP, falling back to greedy")
            return self.greedy_assignment(constraints)
        
        # DP state: (layer_index, accuracy_used)
        # accuracy_used is discretized to integer steps of 0.1%
        max_accuracy_steps = int(constraints.max_accuracy_drop * 10)
        
        # dp[i][acc] = (min_memory, best_assignment)
        dp = {}
        
        def get_layer_cost(layer_name: str, precision: PrecisionType) -> Tuple[float, float]:
            """Get memory cost and accuracy cost for layer at given precision."""
            layer_info = self.layer_info[layer_name]
            sensitivity = layer_info['sensitivity']
            
            if precision == PrecisionType.FP32:
                memory_cost = layer_info['fp32_size_mb']
                accuracy_cost = 0.0
            elif precision == PrecisionType.FP16:
                memory_cost = layer_info['fp16_size_mb']
                accuracy_cost = sensitivity * 0.1
            else:  # INT8
                memory_cost = layer_info['int8_size_mb']
                accuracy_cost = sensitivity
            
            return memory_cost, accuracy_cost
        
        # Initialize DP
        dp[(0, 0)] = (0.0, {})
        
        # Fill DP table
        for layer_idx in range(n_layers):
            layer_name = layers[layer_idx]
            new_dp = {}
            
            for (prev_layer, prev_acc), (prev_memory, prev_assignment) in dp.items():
                if prev_layer != layer_idx:
                    continue
                
                # Try each precision for current layer
                for precision in PrecisionType:
                    memory_cost, accuracy_cost = get_layer_cost(layer_name, precision)
                    
                    new_acc = prev_acc + int(accuracy_cost * 10)
                    new_memory = prev_memory + memory_cost
                    
                    if new_acc <= max_accuracy_steps:
                        state_key = (layer_idx + 1, new_acc)
                        
                        if state_key not in new_dp or new_memory < new_dp[state_key][0]:
                            new_assignment = prev_assignment.copy()
                            new_assignment[layer_name] = precision
                            new_dp[state_key] = (new_memory, new_assignment)
            
            dp.update(new_dp)
        
        # Find best solution
        best_memory = float('inf')
        best_assignment = {}
        
        for (layer_idx, acc), (memory, assignment) in dp.items():
            if layer_idx == n_layers and memory < best_memory:
                best_memory = memory
                best_assignment = assignment
        
        if not best_assignment:
            print("DP failed, falling back to greedy")
            return self.greedy_assignment(constraints)
        
        print(f"DP optimization complete:")
        print(f"  - Optimal memory usage: {best_memory:.2f} MB")
        
        return best_assignment
    
    def evolutionary_assignment(
        self,
        constraints: PrecisionConstraints,
        population_size: int = 50,
        generations: int = 100
    ) -> Dict[str, PrecisionType]:
        """
        Evolutionary algorithm for precision assignment.
        
        Uses genetic algorithm to explore the search space efficiently.
        
        Args:
            constraints: Optimization constraints
            population_size: Size of population for EA
            generations: Number of generations to evolve
            
        Returns:
            Evolved precision assignment
        """
        print(f"Running evolutionary optimization ({population_size} pop, {generations} gen)...")
        
        layers = list(self.sensitivity_scores.keys())
        precisions = list(PrecisionType)
        
        def random_assignment():
            """Generate random precision assignment."""
            return {
                layer: np.random.choice(precisions)
                for layer in layers
            }
        
        def evaluate_fitness(assignment):
            """Evaluate fitness of assignment (lower is better)."""
            total_memory = 0.0
            total_accuracy_drop = 0.0
            
            for layer_name, precision in assignment.items():
                layer_info = self.layer_info[layer_name]
                sensitivity = layer_info['sensitivity']
                
                if precision == PrecisionType.FP32:
                    memory = layer_info['fp32_size_mb']
                    acc_drop = 0.0
                elif precision == PrecisionType.FP16:
                    memory = layer_info['fp16_size_mb']
                    acc_drop = sensitivity * 0.1
                else:  # INT8
                    memory = layer_info['int8_size_mb']
                    acc_drop = sensitivity
                
                total_memory += memory
                total_accuracy_drop += acc_drop
            
            # Penalty for constraint violations
            penalty = 0.0
            if total_accuracy_drop > constraints.max_accuracy_drop:
                penalty += 1000 * (total_accuracy_drop - constraints.max_accuracy_drop)
            
            # Fitness = memory usage + penalty (minimize both)
            return total_memory + penalty
        
        def crossover(parent1, parent2):
            """Crossover two assignments."""
            child = {}
            for layer in layers:
                child[layer] = np.random.choice([parent1[layer], parent2[layer]])
            return child
        
        def mutate(assignment, mutation_rate=0.1):
            """Mutate assignment."""
            mutated = assignment.copy()
            for layer in layers:
                if np.random.random() < mutation_rate:
                    mutated[layer] = np.random.choice(precisions)
            return mutated
        
        # Initialize population
        population = [random_assignment() for _ in range(population_size)]
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [evaluate_fitness(ind) for ind in population]
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 5
                tournament_indices = np.random.choice(
                    len(population), tournament_size, replace=False
                )
                winner_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
                
                if np.random.random() < 0.8:  # Crossover
                    parent2_idx = np.random.choice(len(population))
                    child = crossover(population[winner_idx], population[parent2_idx])
                    child = mutate(child)
                    new_population.append(child)
                else:  # Direct copy
                    new_population.append(mutate(population[winner_idx]))
            
            population = new_population
            
            if generation % 20 == 0:
                best_fitness = min(fitness_scores)
                print(f"  Generation {generation}: Best fitness = {best_fitness:.2f}")
        
        # Return best solution
        final_fitness = [evaluate_fitness(ind) for ind in population]
        best_idx = np.argmin(final_fitness)
        best_assignment = population[best_idx]
        
        print(f"EA optimization complete: Final fitness = {min(final_fitness):.2f}")
        
        return best_assignment
    
    def optimize_precision_assignment(
        self,
        constraints: PrecisionConstraints,
        method: str = "greedy"
    ) -> Dict[str, LayerPrecisionAssignment]:
        """
        Optimize precision assignment using specified method.
        
        Args:
            constraints: Optimization constraints
            method: Optimization method ("greedy", "dp", "evolutionary")
            
        Returns:
            Optimized layer precision assignments
        """
        print(f"Optimizing mixed precision assignment using {method} method...")
        
        if method == "greedy":
            assignment = self.greedy_assignment(constraints)
        elif method == "dp":
            assignment = self.dynamic_programming_assignment(constraints)
        elif method == "evolutionary":
            assignment = self.evolutionary_assignment(constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Convert to detailed assignment objects
        detailed_assignment = {}
        for layer_name, precision in assignment.items():
            layer_info = self.layer_info[layer_name]
            
            if precision == PrecisionType.FP32:
                size_mb = layer_info['fp32_size_mb']
                latency_factor = 1.0
            elif precision == PrecisionType.FP16:
                size_mb = layer_info['fp16_size_mb']
                latency_factor = 0.67
            else:  # INT8
                size_mb = layer_info['int8_size_mb']
                latency_factor = 0.33
            
            detailed_assignment[layer_name] = LayerPrecisionAssignment(
                layer_name=layer_name,
                precision=precision,
                sensitivity_score=layer_info['sensitivity'],
                size_mb=size_mb,
                estimated_latency_ms=latency_factor * 1.0  # Normalized latency
            )
        
        self.optimal_assignment = detailed_assignment
        return detailed_assignment
    
    def analyze_assignment_impact(
        self,
        assignment: Dict[str, LayerPrecisionAssignment]
    ) -> Dict[str, float]:
        """
        Analyze the impact of precision assignment.
        
        Args:
            assignment: Layer precision assignments
            
        Returns:
            Dictionary with impact analysis
        """
        total_layers = len(assignment)
        precision_counts = {p.value: 0 for p in PrecisionType}
        total_size_mb = 0.0
        estimated_accuracy_drop = 0.0
        
        for layer_assignment in assignment.values():
            precision_counts[layer_assignment.precision.value] += 1
            total_size_mb += layer_assignment.size_mb
            
            # Estimate accuracy drop
            if layer_assignment.precision == PrecisionType.FP16:
                estimated_accuracy_drop += layer_assignment.sensitivity_score * 0.1
            elif layer_assignment.precision == PrecisionType.INT8:
                estimated_accuracy_drop += layer_assignment.sensitivity_score
        
        # Calculate original FP32 size
        original_size_mb = sum(
            self.layer_info[name]['fp32_size_mb']
            for name in assignment.keys()
        )
        
        compression_ratio = original_size_mb / total_size_mb if total_size_mb > 0 else 1.0
        
        analysis = {
            'total_layers': total_layers,
            'fp32_layers': precision_counts['fp32'],
            'fp16_layers': precision_counts['fp16'],
            'int8_layers': precision_counts['int8'],
            'int8_ratio': precision_counts['int8'] / total_layers,
            'original_size_mb': original_size_mb,
            'optimized_size_mb': total_size_mb,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - 1/compression_ratio) * 100,
            'estimated_accuracy_drop': estimated_accuracy_drop
        }
        
        print(f"\nMixed Precision Analysis:")
        print(f"  Precision distribution:")
        print(f"    - FP32: {analysis['fp32_layers']} layers")
        print(f"    - FP16: {analysis['fp16_layers']} layers")
        print(f"    - INT8: {analysis['int8_layers']} layers ({analysis['int8_ratio']:.1%})")
        print(f"  Model size: {analysis['original_size_mb']:.1f} MB → {analysis['optimized_size_mb']:.1f} MB")
        print(f"  Compression: {analysis['compression_ratio']:.1f}x ({analysis['size_reduction_percent']:.1f}% reduction)")
        print(f"  Est. accuracy drop: {analysis['estimated_accuracy_drop']:.2f}%")
        
        return analysis


def optimize_mixed_precision(
    model: nn.Module,
    sensitivity_scores: Dict[str, float],
    constraints: Optional[PrecisionConstraints] = None,
    method: str = "greedy"
) -> Tuple[Dict[str, LayerPrecisionAssignment], Dict[str, float]]:
    """
    Convenience function for mixed precision optimization.
    
    Args:
        model: Model to optimize
        sensitivity_scores: Layer sensitivity scores
        constraints: Optimization constraints
        method: Optimization method
        
    Returns:
        Tuple of (precision_assignment, impact_analysis)
    """
    if constraints is None:
        constraints = PrecisionConstraints()
    
    optimizer = MixedPrecisionOptimizer(model, sensitivity_scores)
    assignment = optimizer.optimize_precision_assignment(constraints, method)
    analysis = optimizer.analyze_assignment_impact(assignment)
    
    return assignment, analysis


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    from sensitivity_analysis import analyze_model_sensitivity
    from calibration_dataset import create_calibration_dataloader
    
    # Load model
    model = models.resnet50(pretrained=True)
    
    # Mock sensitivity scores for demo
    sensitivity_scores = {
        'conv1': 2.1,
        'layer1.0.conv1': 0.3,
        'layer1.0.conv2': 0.2,
        'layer2.0.conv1': 0.8,
        'layer2.0.conv2': 0.4,
        'layer3.0.conv1': 1.2,
        'layer3.0.conv2': 0.6,
        'layer4.0.conv1': 1.5,
        'layer4.0.conv2': 0.9,
        'fc': 1.8
    }
    
    # Define constraints
    constraints = PrecisionConstraints(
        max_accuracy_drop=1.0,
        target_compression_ratio=3.0,
        min_int8_ratio=0.6
    )
    
    # Optimize precision assignment
    assignment, analysis = optimize_mixed_precision(
        model=model,
        sensitivity_scores=sensitivity_scores,
        constraints=constraints,
        method="greedy"
    )
    
    print(f"\nOptimization complete!")
    print(f"Achieved {analysis['compression_ratio']:.1f}x compression")
    print(f"Estimated accuracy drop: {analysis['estimated_accuracy_drop']:.2f}%")