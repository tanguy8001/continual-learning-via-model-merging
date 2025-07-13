#!/usr/bin/env python3
"""
Add loss landscape visualization to the existing curve merging test.

This script modifies the test_curve_merging_mlp.py to include loss landscape visualization
using PCA components from the learned curve.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from curves_MLP import sample_curve_points, analyze_curve_dimensionality


def compute_pca_from_curve(curve_mlp, model_A, model_B, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA components from the learned curve between two models.
    
    Args:
        curve_mlp: The trained curve MLP
        model_A: First model
        model_B: Second model
        num_points: Number of points to sample along the curve
        
    Returns:
        Tuple of (pca_components, explained_variance_ratio)
    """
    print("Computing PCA components from learned curve...")
    
    # Get flattened weights
    flat_A = torch.cat([p.view(-1) for p in model_A.parameters()])
    flat_B = torch.cat([p.view(-1) for p in model_B.parameters()])
    
    # Sample points along the curve
    X = sample_curve_points(curve_mlp, flat_A, flat_B, num_points=num_points)
    
    # Compute PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    print(f"PCA computed. Explained variance: {pca.explained_variance_ratio_}")
    
    return pca.components_, pca.explained_variance_ratio_


def compute_loss_at_point(model_A, pca_components, x_coord: float, y_coord: float, 
                         test_loader, device: str) -> float:
    """
    Compute loss at a specific point in PCA space.
    
    Args:
        model_A: Base model (model A)
        pca_components: PCA components from the curve
        x_coord: X coordinate in PCA space
        y_coord: Y coordinate in PCA space
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        Loss value at the specified point
    """
    # Get base weights (model A)
    base_weights = torch.cat([p.view(-1) for p in model_A.parameters()])
    
    # Compute new weights using PCA directions
    pc1 = torch.from_numpy(pca_components[0]).to(device)
    pc2 = torch.from_numpy(pca_components[1]).to(device)
    new_weights = base_weights + x_coord * pc1 + y_coord * pc2
    
    # Create temporary model with new weights
    model_class = type(model_A)
    temp_model = model_class(
        input_dim=784,
        num_classes=10,
        hidden_dims=[400, 200, 100]
    ).to(device)
    
    # Set weights
    start_idx = 0
    for param in temp_model.parameters():
        end_idx = start_idx + param.numel()
        param.data = new_weights[start_idx:end_idx].view_as(param.data)
        start_idx = end_idx
        
    # Compute loss
    temp_model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = temp_model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
    return total_loss / total_samples


def create_loss_landscape(curve_mlp, model_A, model_B, test_loader, device: str,
                         x_range: Tuple[float, float] = (-0.5, 0.5), 
                         y_range: Tuple[float, float] = (-0.5, 0.5), 
                         resolution: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a loss landscape grid using PCA components from the learned curve.
    
    Args:
        curve_mlp: The trained curve MLP
        model_A: First model
        model_B: Second model
        test_loader: Test data loader
        device: Device to use
        x_range: Range for x-coordinate
        y_range: Range for y-coordinate
        resolution: Resolution of the grid
        
    Returns:
        Tuple of (X, Y, loss_matrix) for plotting
    """
    print(f"Computing loss landscape with resolution {resolution}x{resolution}...")
    
    # Compute PCA components
    pca_components, explained_variance = compute_pca_from_curve(curve_mlp, model_A, model_B, num_points=500)
    
    x_coords = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    loss_matrix = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            loss = compute_loss_at_point(model_A, pca_components, x_coords[i], y_coords[j], test_loader, device)
            loss_matrix[j, i] = loss
            print(f"Progress: {i*resolution + j + 1}/{resolution*resolution} - Loss: {loss:.4f}")
            
    return X, Y, loss_matrix, pca_components, explained_variance


def plot_loss_landscape_with_curve(X: np.ndarray, Y: np.ndarray, loss_matrix: np.ndarray,
                                 curve_mlp, model_A, model_B, pca_components,
                                 save_path: str = './curve_loss_landscape.png') -> None:
    """
    Plot the loss landscape with the learned curve trajectory.
    
    Args:
        X: X coordinate meshgrid
        Y: Y coordinate meshgrid
        loss_matrix: Loss values matrix
        curve_mlp: The trained curve MLP
        model_A: First model
        model_B: Second model
        pca_components: PCA components
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create filled contour plot
    filled_contour = plt.contourf(X, Y, loss_matrix, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(filled_contour, label='Loss')
    
    # Add contour lines
    contour = plt.contour(X, Y, loss_matrix, levels=20, colors='black', alpha=0.6, linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Add curve trajectory
    _plot_curve_trajectory(curve_mlp, model_A, model_B, pca_components)
    
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.title('Loss Landscape using Learned Curve PCA Components')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss landscape plot saved to {save_path}")
    plt.show()


def _plot_curve_trajectory(curve_mlp, model_A, model_B, pca_components):
    """Plot the learned curve trajectory on the loss landscape."""
    # Sample points along the curve
    flat_A = torch.cat([p.view(-1) for p in model_A.parameters()])
    flat_B = torch.cat([p.view(-1) for p in model_B.parameters()])
    
    t_values = np.linspace(0, 1, 50)
    curve_points = []
    
    for t in t_values:
        t_tensor = torch.tensor(t, dtype=flat_A.dtype, device=flat_A.device)
        w_curve = curve_mlp(t_tensor, flat_A, flat_B).detach()
        
        # Project onto PCA directions
        base_weights = torch.cat([p.view(-1) for p in model_A.parameters()])
        diff = w_curve - base_weights
        
        pc1 = torch.from_numpy(pca_components[0]).to(flat_A.device)
        pc2 = torch.from_numpy(pca_components[1]).to(flat_A.device)
        
        x_proj = torch.dot(diff, pc1).item()
        y_proj = torch.dot(diff, pc2).item()
        
        curve_points.append([x_proj, y_proj])
    
    curve_points = np.array(curve_points)
    
    # Plot curve trajectory
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', linewidth=3, label='Learned Curve', alpha=0.8)
    plt.plot(curve_points[0, 0], curve_points[0, 1], 'go', markersize=10, label='Model A (t=0)')
    plt.plot(curve_points[-1, 0], curve_points[-1, 1], 'bo', markersize=10, label='Model B (t=1)')
    plt.plot(curve_points[25, 0], curve_points[25, 1], 'mo', markersize=8, label='Midpoint (t=0.5)')
    
    plt.legend()


def add_loss_landscape_to_test():
    """
    Function to be called from test_curve_merging_mlp.py to add loss landscape visualization.
    
    This function should be called after the curve MLP has been trained and evaluated.
    """
    print("\n" + "="*50)
    print("ADDING LOSS LANDSCAPE VISUALIZATION")
    print("="*50)
    
    # This function assumes that the following variables are available in the global scope
    # from test_curve_merging_mlp.py:
    # - curve_mlp: The trained curve MLP
    # - model_A: First model
    # - model_B: Second model
    # - test_loader: Test data loader
    # - device: Device being used
    
    # You would call this function at the end of test_curve_merging_mlp.py like this:
    # add_loss_landscape_to_test()
    
    print("Loss landscape visualization function ready to be integrated!")
    print("To use this:")
    print("1. Import this function in test_curve_merging_mlp.py")
    print("2. Call it after the curve MLP training is complete")
    print("3. Pass the required variables (curve_mlp, model_A, model_B, test_loader, device)")


# Example of how to integrate this with test_curve_merging_mlp.py:
def example_integration():
    """
    Example of how to integrate loss landscape visualization with test_curve_merging_mlp.py
    """
    print("Example integration with test_curve_merging_mlp.py:")
    print()
    print("1. Add this import at the top of test_curve_merging_mlp.py:")
    print("   from add_loss_landscape_to_test import create_loss_landscape, plot_loss_landscape_with_curve")
    print()
    print("2. Add this code at the end of the main() function, after curve training:")
    print("""
    # Add loss landscape visualization
    print("\\nCreating loss landscape visualization...")
    X, Y, loss_matrix, pca_components, explained_variance = create_loss_landscape(
        curve_mlp, model_A, model_B, test_loader, device,
        x_range=(-0.3, 0.3), 
        y_range=(-0.3, 0.3), 
        resolution=15  # Lower resolution for faster computation
    )
    
    plot_loss_landscape_with_curve(X, Y, loss_matrix, curve_mlp, model_A, model_B, pca_components)
    
    print(f"\\nLoss landscape analysis complete!")
    print(f"1st PC explains {explained_variance[0]*100:.1f}% of variance")
    print(f"2nd PC explains {explained_variance[1]*100:.1f}% of variance")
    print(f"Total explained by 2 components: {np.sum(explained_variance[:2])*100:.1f}%")
    """)


if __name__ == "__main__":
    example_integration() 