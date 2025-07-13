#!/usr/bin/env python3
"""
Example script demonstrating curve-based loss landscape visualization.

This script shows how to:
1. Use the existing curve merging setup from test_curve_merging_mlp.py
2. Extract PCA components from the learned curve
3. Create loss landscape visualizations

Usage:
    python example_curve_loss_landscape.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import h5py
from typing import Tuple, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_curve_merging_mlp import main as run_curve_merging
from curves_MLP import CurveMLP, sample_curve_points, analyze_curve_dimensionality
from models.mlpnet import MlpNetBase
from datasets import HeteroMNIST


class SimpleCurveLossLandscape:
    """
    A simplified version of curve-based loss landscape visualization.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model_A: Optional[nn.Module] = None
        self.model_B: Optional[nn.Module] = None
        self.curve_mlp: Optional[CurveMLP] = None
        self.test_loader: Optional[DataLoader] = None
        self.pca_components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        
    def load_models_from_checkpoints(self, model_A_path: str, model_B_path: str) -> None:
        """Load models from checkpoint files."""
        # Load model A
        checkpoint_A = torch.load(model_A_path, map_location=self.device)
        self.model_A = MlpNetBase(
            input_dim=checkpoint_A.get('config', {}).get('input_dim', 784),
            num_classes=checkpoint_A.get('config', {}).get('num_classes', 10),
            hidden_dims=checkpoint_A.get('config', {}).get('hidden_dims', [400, 200, 100])
        )
        self.model_A.load_state_dict(checkpoint_A['model_state_dict'])
        self.model_A.to(self.device)
        self.model_A.eval()
        
        # Load model B
        checkpoint_B = torch.load(model_B_path, map_location=self.device)
        self.model_B = MlpNetBase(
            input_dim=checkpoint_B.get('config', {}).get('input_dim', 784),
            num_classes=checkpoint_B.get('config', {}).get('num_classes', 10),
            hidden_dims=checkpoint_B.get('config', {}).get('hidden_dims', [400, 200, 100])
        )
        self.model_B.load_state_dict(checkpoint_B['model_state_dict'])
        self.model_B.to(self.device)
        self.model_B.eval()
        
        print(f"Loaded models from checkpoints")
        
    def create_curve_mlp(self, hidden_dim: int = 32, t_only_mode: bool = False) -> None:
        """Create a curve MLP for interpolation."""
        if self.model_A is None or self.model_B is None:
            raise ValueError("Models must be loaded first")
            
        # Get parameter dimensions
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        self.curve_mlp = CurveMLP(
            in_features=flat_A.numel() * 2,
            out_features=flat_A.numel(),
            bias=True,
            hidden_dim=hidden_dim,
            t_only_mode=t_only_mode
        ).to(self.device)
        
        print(f"Created curve MLP with {flat_A.numel()} parameters")
        
    def setup_data_loader(self, batch_size: int = 128) -> None:
        """Setup test data loader."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = HeteroMNIST(root='./data', train=False, transform=transform, download=True)  # type: ignore
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"Setup data loader for MNIST test set")
        
    def compute_pca_from_curve(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PCA components from the learned curve."""
        if self.model_A is None or self.model_B is None or self.curve_mlp is None:
            raise ValueError("Models and curve MLP must be loaded first")
            
        print("Computing PCA components from curve...")
        
        # Get flattened weights
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        # Sample points along the curve
        X = sample_curve_points(self.curve_mlp, flat_A, flat_B, num_points=num_points)
        
        # Compute PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        
        self.pca_components = pca.components_
        self.explained_variance = pca.explained_variance_ratio_
        
        print(f"PCA computed. Explained variance: {pca.explained_variance_ratio_}")
        
        return pca.components_, pca.explained_variance_ratio_
        
    def compute_loss_at_point(self, x_coord: float, y_coord: float) -> float:
        """Compute loss at a specific point in PCA space."""
        if self.model_A is None or self.pca_components is None or self.test_loader is None:
            raise ValueError("Models, PCA components, and test loader must be loaded first")
            
        # Get base weights (model A)
        base_weights = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        
        # Compute new weights using PCA directions
        pc1 = torch.from_numpy(self.pca_components[0]).to(self.device)
        pc2 = torch.from_numpy(self.pca_components[1]).to(self.device)
        new_weights = base_weights + x_coord * pc1 + y_coord * pc2
        
        # Create temporary model with new weights
        temp_model = MlpNetBase(
            input_dim=784,
            num_classes=10,
            hidden_dims=[400, 200, 100]
        ).to(self.device)
        
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
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = temp_model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                
        return total_loss / total_samples
        
    def create_loss_landscape(self, x_range: Tuple[float, float] = (-0.5, 0.5), 
                            y_range: Tuple[float, float] = (-0.5, 0.5), 
                            resolution: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a loss landscape grid."""
        print(f"Computing loss landscape with resolution {resolution}x{resolution}...")
        
        x_coords = np.linspace(x_range[0], x_range[1], resolution)
        y_coords = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        loss_matrix = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                loss = self.compute_loss_at_point(x_coords[i], y_coords[j])
                loss_matrix[j, i] = loss
                print(f"Progress: {i*resolution + j + 1}/{resolution*resolution} - Loss: {loss:.4f}")
                
        return X, Y, loss_matrix
        
    def plot_loss_landscape(self, X: np.ndarray, Y: np.ndarray, loss_matrix: np.ndarray, 
                          save_path: str = './curve_loss_landscape.png') -> None:
        """Plot the loss landscape with curve trajectory."""
        plt.figure(figsize=(12, 10))
        
        # Create filled contour plot
        filled_contour = plt.contourf(X, Y, loss_matrix, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(filled_contour, label='Loss')
        
        # Add contour lines
        contour = plt.contour(X, Y, loss_matrix, levels=20, colors='black', alpha=0.6, linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8)
        
        # Add curve trajectory if available
        if self.model_A is not None and self.model_B is not None and self.curve_mlp is not None:
            self._plot_curve_trajectory(X, Y)
        
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.title(f'Loss Landscape using Learned Curve PCA Components\nExplained variance: {self.explained_variance[0]*100:.1f}%, {self.explained_variance[1]*100:.1f}%')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss landscape plot saved to {save_path}")
        plt.show()
        
    def _plot_curve_trajectory(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Plot the learned curve trajectory on the loss landscape."""
        if self.model_A is None or self.model_B is None or self.curve_mlp is None or self.pca_components is None:
            return
            
        # Sample points along the curve
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        t_values = np.linspace(0, 1, 50)
        curve_points = []
        
        for t in t_values:
            t_tensor = torch.tensor(t, dtype=flat_A.dtype, device=flat_A.device)
            w_curve = self.curve_mlp(t_tensor, flat_A, flat_B).detach()
            
            # Project onto PCA directions
            base_weights = torch.cat([p.view(-1) for p in self.model_A.parameters()])
            diff = w_curve - base_weights
            
            pc1 = torch.from_numpy(self.pca_components[0]).to(self.device)
            pc2 = torch.from_numpy(self.pca_components[1]).to(self.device)
            
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
        
    def plot_curve_dimensionality(self, save_path: str = './curve_dimensionality.png') -> None:
        """Plot the intrinsic dimensionality analysis of the learned curve."""
        if self.model_A is None or self.model_B is None or self.curve_mlp is None:
            raise ValueError("Models and curve MLP must be loaded first")
            
        # Compute curve dimensionality
        n_components, explained_variance = self.curve_mlp.evaluate_curve_dimensionality(
            self.model_A, self.model_B, num_points=1000, plot=False
        )
        
        # Create plot
        plt.figure(figsize=(10, 8))
        cumulative = np.cumsum(explained_variance)
        
        plt.plot(np.arange(1, len(cumulative)+1), cumulative*100, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=99, color='r', linestyle='--', alpha=0.7, label='99% variance threshold')
        plt.axvline(x=n_components, color='g', linestyle='--', alpha=0.7, 
                   label=f'{n_components} components needed')
        
        plt.xlabel('Number of Principal Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance (%)', fontsize=12)
        plt.title('Intrinsic Dimensionality of Learned Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text annotations
        plt.text(0.02, 0.98, f'1st PC: {explained_variance[0]*100:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(0.02, 0.92, f'2nd PC: {explained_variance[1]*100:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.text(0.02, 0.86, f'Total for 2 PCs: {np.sum(explained_variance[:2])*100:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curve dimensionality plot saved to {save_path}")
        plt.show()


def main():
    """Main function to demonstrate curve-based loss landscape visualization."""
    print("Curve-based Loss Landscape Visualization Example")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if models exist
    model_A_path = "checkpoints/cifar10_model_A.pth"
    model_B_path = "checkpoints/cifar10_model_B.pth"
    
    if not (os.path.exists(model_A_path) and os.path.exists(model_B_path)):
        print("Model checkpoints not found. Running curve merging first...")
        # Run the curve merging to create models
        run_curve_merging()
    else:
        print("Using existing model checkpoints...")
    
    # Create visualizer
    visualizer = SimpleCurveLossLandscape(device=device)
    
    # Load models
    visualizer.load_models_from_checkpoints(model_A_path, model_B_path)
    
    # Setup data loader
    visualizer.setup_data_loader()
    
    # Create curve MLP (you would normally train this, but for demo we'll create an untrained one)
    visualizer.create_curve_mlp(hidden_dim=32, t_only_mode=False)
    
    # Compute PCA components from curve
    pca_components, explained_variance = visualizer.compute_pca_from_curve(num_points=500)
    
    print(f"\nPCA Analysis Results:")
    print(f"1st Principal Component explains {explained_variance[0]*100:.1f}% of variance")
    print(f"2nd Principal Component explains {explained_variance[1]*100:.1f}% of variance")
    print(f"Total explained by 2 components: {np.sum(explained_variance[:2])*100:.1f}%")
    
    # Plot curve dimensionality
    visualizer.plot_curve_dimensionality()
    
    # Create and plot loss landscape (with lower resolution for faster computation)
    print("\nCreating loss landscape visualization...")
    X, Y, loss_matrix = visualizer.create_loss_landscape(
        x_range=(-0.3, 0.3), 
        y_range=(-0.3, 0.3), 
        resolution=15  # Lower resolution for faster computation
    )
    
    visualizer.plot_loss_landscape(X, Y, loss_matrix)
    
    print("\nVisualization complete!")
    print("Check the generated plots to see:")
    print("- The intrinsic dimensionality of the learned curve")
    print("- The loss landscape projected onto the PCA components")
    print("- The trajectory of the learned curve through the landscape")


if __name__ == "__main__":
    main() 