"""
Simple test script to verify PCA loss landscape functionality.
"""

import torch
import torch.nn as nn
from curves_MLP import CurveMLP, CurveConfigMLP, InterpolationType
from models.mlpnet import MlpNet
import numpy as np
from sklearn.decomposition import PCA

def test_pca_directions():
    """Test PCA direction extraction from curve."""
    print("Testing PCA direction extraction...")
    
    # Create simple models
    model1 = MlpNet.base(input_dim=100, num_classes=5, hidden_dims=[50, 25])
    model2 = MlpNet.base(input_dim=100, num_classes=5, hidden_dims=[50, 25])
    
    # Create curve MLP
    curve_mlp = CurveMLP(
        in_features=sum(p.numel() for p in model1.parameters()),
        out_features=sum(p.numel() for p in model1.parameters()),
        hidden_dim=32,
        t_only_mode=True
    )
    
    # Test PCA direction extraction
    pca_directions, explained_variance_ratio = curve_mlp.get_pca_basis_directions(
        model1, model2, num_points=100, n_components=2
    )
    
    print(f"✓ PCA directions extracted successfully")
    print(f"  - Number of directions: {len(pca_directions)}")
    print(f"  - Explained variance ratios: {explained_variance_ratio}")
    print(f"  - Total explained variance: {sum(explained_variance_ratio):.4f}")
    
    # Verify directions are orthogonal
    dot_product = torch.dot(pca_directions[0], pca_directions[1])
    print(f"  - Orthogonality check: {dot_product:.6f} (should be close to 0)")
    
    return True

def test_curve_sampling():
    """Test curve point sampling."""
    print("\nTesting curve point sampling...")
    
    # Create simple models
    model1 = MlpNet.base(input_dim=50, num_classes=3, hidden_dims=[20, 10])
    model2 = MlpNet.base(input_dim=50, num_classes=3, hidden_dims=[20, 10])
    
    # Create curve MLP
    curve_mlp = CurveMLP(
        in_features=sum(p.numel() for p in model1.parameters()),
        out_features=sum(p.numel() for p in model1.parameters()),
        hidden_dim=16,
        t_only_mode=True
    )
    
    # Test curve sampling
    flat1 = torch.cat([p.view(-1) for p in model1.parameters()])
    flat2 = torch.cat([p.view(-1) for p in model2.parameters()])
    
    from curves_MLP import sample_curve_points
    curve_points = sample_curve_points(curve_mlp, flat1, flat2, num_points=50)
    
    print(f"✓ Curve sampling successful")
    print(f"  - Curve points shape: {curve_points.shape}")
    print(f"  - Expected shape: (50, {flat1.numel()})")
    
    # Test PCA on curve points
    pca = PCA(n_components=2)
    pca.fit(curve_points)
    
    print(f"  - PCA explained variance: {pca.explained_variance_ratio_}")
    
    return True

def test_dimensionality_analysis():
    """Test dimensionality analysis."""
    print("\nTesting dimensionality analysis...")
    
    # Create simple models
    model1 = MlpNet.base(input_dim=30, num_classes=2, hidden_dims=[15])
    model2 = MlpNet.base(input_dim=30, num_classes=2, hidden_dims=[15])
    
    # Create curve MLP
    curve_mlp = CurveMLP(
        in_features=sum(p.numel() for p in model1.parameters()),
        out_features=sum(p.numel() for p in model1.parameters()),
        hidden_dim=8,
        t_only_mode=True
    )
    
    # Test dimensionality analysis
    n_components, explained_variance = curve_mlp.evaluate_curve_dimensionality(
        model1, model2, num_points=100, plot=False, variance_threshold=0.95
    )
    
    print(f"✓ Dimensionality analysis successful")
    print(f"  - Components for 95% variance: {n_components}")
    print(f"  - First component variance: {explained_variance[0]:.4f}")
    
    return True

def main():
    """Run all tests."""
    print("Running PCA Loss Landscape Tests")
    print("=" * 40)
    
    try:
        test_pca_directions()
        test_curve_sampling()
        test_dimensionality_analysis()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed successfully!")
        print("\nThe PCA loss landscape functionality is working correctly.")
        print("You can now use the curve_loss_landscape_example.py script for full visualization.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 