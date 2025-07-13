#!/usr/bin/env python3
"""
Simple script to run loss landscape visualization with existing curve merging setup.

This script demonstrates how to add loss landscape visualization to your existing
curve merging experiment.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from curves_MLP import sample_curve_points
from models.mlpnet import MlpNetBase


def compute_pca_from_curve(curve_mlp, model_A, model_B, num_points: int = 500):
    """Compute PCA components from the learned curve."""
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
    """Compute loss at a specific point in PCA space."""
    # Get base weights (model A)
    base_weights = torch.cat([p.view(-1) for p in model_A.parameters()])
    
    # Compute new weights using PCA directions
    pc1 = torch.from_numpy(pca_components[0]).to(device)
    pc2 = torch.from_numpy(pca_components[1]).to(device)
    new_weights = base_weights + x_coord * pc1 + y_coord * pc2
    
    # Create temporary model with new weights
    temp_model = MlpNetBase(
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
                         x_range: Tuple[float, float] = (-0.3, 0.3), 
                         y_range: Tuple[float, float] = (-0.3, 0.3), 
                         resolution: int = 15):
    """Create a loss landscape grid using PCA components from the learned curve."""
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


def plot_loss_landscape_with_curve(X, Y, loss_matrix, curve_mlp, model_A, model_B, pca_components):
    """Plot the loss landscape with the learned curve trajectory."""
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
    
    plt.savefig('./curve_loss_landscape.png', dpi=300, bbox_inches='tight')
    print("Loss landscape plot saved to ./curve_loss_landscape.png")
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


def main():
    """Main function to demonstrate loss landscape visualization."""
    print("Loss Landscape Visualization Demo")
    print("=" * 40)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if models exist
    model_A_path = "checkpoints/cifar10_model_A.pth"
    model_B_path = "checkpoints/cifar10_model_B.pth"
    
    if not (os.path.exists(model_A_path) and os.path.exists(model_B_path)):
        print("Model checkpoints not found!")
        print("Please run test_curve_merging_mlp.py first to create the models.")
        return
    
    print("Loading models from checkpoints...")
    
    # Load models
    checkpoint_A = torch.load(model_A_path, map_location=device)
    model_A = MlpNetBase(
        input_dim=checkpoint_A.get('config', {}).get('input_dim', 784),
        num_classes=checkpoint_A.get('config', {}).get('num_classes', 10),
        hidden_dims=checkpoint_A.get('config', {}).get('hidden_dims', [400, 200, 100])
    )
    model_A.load_state_dict(checkpoint_A['model_state_dict'])
    model_A.to(device)
    model_A.eval()
    
    checkpoint_B = torch.load(model_B_path, map_location=device)
    model_B = MlpNetBase(
        input_dim=checkpoint_B.get('config', {}).get('input_dim', 784),
        num_classes=checkpoint_B.get('config', {}).get('num_classes', 10),
        hidden_dims=checkpoint_B.get('config', {}).get('hidden_dims', [400, 200, 100])
    )
    model_B.load_state_dict(checkpoint_B['model_state_dict'])
    model_B.to(device)
    model_B.eval()
    
    print("Models loaded successfully!")
    
    # Setup test data loader
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print("Test data loader setup complete!")
    
    # Create a simple curve MLP (untrained for demo)
    flat_A = torch.cat([p.view(-1) for p in model_A.parameters()])
    flat_B = torch.cat([p.view(-1) for p in model_B.parameters()])
    
    from curves_MLP import CurveMLP
    curve_mlp = CurveMLP(
        in_features=flat_A.numel() * 2,
        out_features=flat_A.numel(),
        bias=True,
        hidden_dim=32,
        t_only_mode=False
    ).to(device)
    
    print(f"Created curve MLP with {flat_A.numel()} parameters")
    
    # Create and plot loss landscape
    print("\nCreating loss landscape visualization...")
    X, Y, loss_matrix, pca_components, explained_variance = create_loss_landscape(
        curve_mlp, model_A, model_B, test_loader, device,
        x_range=(-0.3, 0.3), 
        y_range=(-0.3, 0.3), 
        resolution=15  # Lower resolution for faster computation
    )
    
    plot_loss_landscape_with_curve(X, Y, loss_matrix, curve_mlp, model_A, model_B, pca_components)
    
    print(f"\nLoss landscape analysis complete!")
    print(f"1st PC explains {explained_variance[0]*100:.1f}% of variance")
    print(f"2nd PC explains {explained_variance[1]*100:.1f}% of variance")
    print(f"Total explained by 2 components: {np.sum(explained_variance[:2])*100:.1f}%")
    
    print("\nTo integrate this with your existing test_curve_merging_mlp.py:")
    print("1. Import the functions from this script")
    print("2. Call create_loss_landscape() after training the curve MLP")
    print("3. Call plot_loss_landscape_with_curve() to visualize the results")


if __name__ == "__main__":
    main() 