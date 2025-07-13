#!/usr/bin/env python3
"""
Test script to demonstrate the difference in intrinsic dimensionality 
between different curve parametrizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from curves_MLP import CurveMLP, CurveConfigMLP, InterpolationType, analyze_curve_dimensionality

def create_simple_model():
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

def test_dimensionality_comparison():
    """Compare intrinsic dimensionality of different curve parametrizations."""
    
    # Create two simple models
    model1 = create_simple_model()
    model2 = create_simple_model()
    
    # Get flattened weights
    flat1 = torch.cat([p.view(-1) for p in model1.parameters()])
    flat2 = torch.cat([p.view(-1) for p in model2.parameters()])
    
    print(f"Model parameter count: {flat1.shape[0]}")
    
    # Test different parametrizations
    configs = [
        ("Original (Quadratic)", {"num_terms": 1, "use_fourier_basis": False}),
        ("Polynomial (3 terms)", {"num_terms": 3, "use_fourier_basis": False}),
        ("Fourier (6 terms)", {"num_terms": 6, "use_fourier_basis": True}),
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\n=== Testing {name} ===")
        
        # Create curve with specified config
        curve = CurveMLP(
            in_features=flat1.shape[0],
            out_features=flat1.shape[0],
            hidden_dim=64,
            t_only_mode=True,
            num_terms=config["num_terms"]
        )
        
        # Sample points along the curve
        t_values = np.linspace(0, 1, 1000)
        curve_points = []
        
        for t in t_values:
            t_tensor = torch.tensor(t, dtype=flat1.dtype)
            if config["use_fourier_basis"]:
                w_curve = curve.forward_fourier(t_tensor, flat1, flat2, num_fourier_terms=config["num_terms"])
            else:
                w_curve = curve(t_tensor, flat1, flat2)
            curve_points.append(w_curve.detach().numpy())
        
        X = np.stack(curve_points)
        
        # Analyze dimensionality
        n_components, explained_variance = analyze_curve_dimensionality(
            X, plot=False, variance_threshold=0.99
        )
        
        results[name] = {
            "n_components": n_components,
            "explained_variance": explained_variance,
            "first_component": explained_variance[0],
            "first_two_components": np.sum(explained_variance[:2])
        }
        
        print(f"Components to explain 99% variance: {n_components}")
        print(f"First component explains: {explained_variance[0]:.4f}")
        print(f"First two components explain: {np.sum(explained_variance[:2]):.4f}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative explained variance
    for name, result in results.items():
        cumulative = np.cumsum(result["explained_variance"])
        ax1.plot(np.arange(1, len(cumulative)+1), cumulative*100, marker='o', label=name)
    
    ax1.set_xlabel('Number of principal components')
    ax1.set_ylabel('Cumulative explained variance (%)')
    ax1.set_title('Curve Dimensionality Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot bar chart of components needed
    names = list(results.keys())
    n_components = [results[name]["n_components"] for name in names]
    ax2.bar(names, n_components)
    ax2.set_ylabel('Components for 99% variance')
    ax2.set_title('Intrinsic Dimensionality')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('curve_dimensionality_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = test_dimensionality_comparison()
    
    print("\n=== Summary ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Components for 99% variance: {result['n_components']}")
        print(f"  First component: {result['first_component']:.4f}")
        print(f"  First two components: {result['first_two_components']:.4f}")
        print() 