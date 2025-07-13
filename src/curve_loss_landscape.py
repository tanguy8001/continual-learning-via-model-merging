#!/usr/bin/env python3
"""
Loss Landscape Visualization using PCA Components from Learned Curve

This script visualizes the loss landscape by:
1. Using the PCA components computed from the learned curve between two models
2. Projecting the loss surface onto the 2D plane spanned by the first two principal components
3. Creating contour plots, 3D surfaces, and trajectory visualizations

Usage:
    python curve_loss_landscape.py --model_A_path checkpoints/model_A.pth --model_B_path checkpoints/model_B.pth --curve_mlp_path checkpoints/curve_mlp.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import warnings
from typing import List, Tuple, Optional, Any, Union
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict
warnings.filterwarnings('ignore')

# Add the src directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fcmodel import FCModel
from models.mlpnet import MlpNetBase
from curves_MLP import CurveMLP, sample_curve_points, analyze_curve_dimensionality
from datasets import HeteroMNIST
import data


class CurveLossLandscapeVisualizer:
    """
    A class for visualizing loss landscapes using PCA components from learned curves.
    """
    
    def __init__(self, model_name: str, dataset_name: str, device: str = 'cpu'):
        """
        Initialize the visualizer.
        
        Args:
            model_name: Name of the model architecture ('FC', 'MlpNet', etc.)
            dataset_name: Name of the dataset ('MNIST', 'CIFAR10', etc.)
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device
        self.model_A: Optional[nn.Module] = None
        self.model_B: Optional[nn.Module] = None
        self.curve_mlp: Optional[CurveMLP] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.pca_components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        
    def load_models_and_curve(self, model_A_path: str, model_B_path: str, curve_mlp_path: str) -> None:
        """
        Load trained models and the learned curve MLP.
        
        Args:
            model_A_path: Path to model A checkpoint
            model_B_path: Path to model B checkpoint
            curve_mlp_path: Path to the trained curve MLP
        """
        if not all(os.path.exists(path) for path in [model_A_path, model_B_path, curve_mlp_path]):
            raise FileNotFoundError(f"One or more model files not found")
            
        # Load model A
        checkpoint_A = torch.load(model_A_path, map_location=self.device)
        if self.model_name == 'FC':
            config = checkpoint_A.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
            self.model_A = FCModel.base(**config)
        elif self.model_name == 'MlpNet':
            config = checkpoint_A.get('config', {'input_dim': 784, 'num_classes': 10})
            self.model_A = MlpNetBase(**config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        self.model_A.load_state_dict(checkpoint_A['model_state_dict'])
        self.model_A.to(self.device)
        self.model_A.eval()
        
        # Load model B
        checkpoint_B = torch.load(model_B_path, map_location=self.device)
        if self.model_name == 'FC':
            config = checkpoint_B.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
            self.model_B = FCModel.base(**config)
        elif self.model_name == 'MlpNet':
            config = checkpoint_B.get('config', {'input_dim': 784, 'num_classes': 10})
            self.model_B = MlpNetBase(**config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        self.model_B.load_state_dict(checkpoint_B['model_state_dict'])
        self.model_B.to(self.device)
        self.model_B.eval()
        
        # Load curve MLP
        curve_checkpoint = torch.load(curve_mlp_path, map_location=self.device)
        curve_config = curve_checkpoint.get('config', {})
        
        # Get parameter dimensions
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        self.curve_mlp = CurveMLP(
            in_features=flat_A.numel() * 2,
            out_features=flat_A.numel(),
            bias=True,
            hidden_dim=curve_config.get('hidden_dim', 32),
            t_only_mode=curve_config.get('t_only_mode', False)
        )
        self.curve_mlp.load_state_dict(curve_checkpoint['curve_state_dict'])
        self.curve_mlp.to(self.device)
        self.curve_mlp.eval()
        
        print(f"Loaded models and curve MLP")
        print(f"Model A accuracy: {checkpoint_A.get('val_acc', 'N/A')}%")
        print(f"Model B accuracy: {checkpoint_B.get('val_acc', 'N/A')}%")
        
    def setup_data_loaders(self, data_path: str = './data', batch_size: int = 128) -> None:
        """
        Setup data loaders for evaluation.
        
        Args:
            data_path: Path to the dataset
            batch_size: Batch size for data loading
        """
        if self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            test_dataset = HeteroMNIST(root=data_path, train=False, transform=transform, download=True)  # type: ignore
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # type: ignore
            
        elif self.dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        print(f"Setup data loaders for {self.dataset_name}")
        
    def compute_pca_components_from_curve(self, num_points: int = 1000, save_dir: str = './pca_components') -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Compute PCA components from the learned curve between two models.
        
        Args:
            num_points: Number of points to sample along the curve
            save_dir: Directory to save PCA components
            
        Returns:
            Tuple of (x_direction, y_direction, explained_variance_ratio)
        """
        os.makedirs(save_dir, exist_ok=True)
        dir_file = os.path.join(save_dir, 'curve_pca_components.h5')
        
        # Check if PCA components already exist
        if os.path.exists(dir_file):
            print(f"Loading existing PCA components from {dir_file}")
            with h5py.File(dir_file, 'r') as f:
                x_direction = torch.from_numpy(np.array(f['xdirection'])).float()
                y_direction = torch.from_numpy(np.array(f['ydirection'])).float()
                explained_variance_ratio = np.array(f['explained_variance_ratio_'])
            return x_direction, y_direction, explained_variance_ratio
        
        print("Computing PCA components from learned curve...")
        
        if self.model_A is None or self.model_B is None or self.curve_mlp is None:
            raise ValueError("Models and curve MLP must be loaded first")
            
        # Get flattened weights
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        # Sample points along the curve
        X = sample_curve_points(self.curve_mlp, flat_A, flat_B, num_points=num_points)
        
        # Compute PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        
        # Get principal components
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        
        # Convert to tensor format for saving
        x_direction = torch.from_numpy(pc1).float()
        y_direction = torch.from_numpy(pc2).float()
        
        # Save PCA components
        with h5py.File(dir_file, 'w') as f:
            f.create_dataset('xdirection', data=x_direction.numpy())
            f.create_dataset('ydirection', data=y_direction.numpy())
            f.create_dataset('explained_variance_ratio_', data=pca.explained_variance_ratio_)
            f.create_dataset('singular_values_', data=pca.singular_values_)
            f.create_dataset('explained_variance_', data=pca.explained_variance_)
            
        print(f"PCA components saved to {dir_file}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        self.pca_components = pca.components_
        self.explained_variance = pca.explained_variance_ratio_
        
        return x_direction, y_direction, pca.explained_variance_ratio_
        
    def get_model_weights(self, model: nn.Module) -> torch.Tensor:
        """
        Get flattened weights from a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Flattened weight tensor
        """
        return torch.cat([p.view(-1) for p in model.parameters()])
        
    def set_model_weights(self, model: nn.Module, weights: torch.Tensor) -> None:
        """
        Set model weights from a flattened tensor.
        
        Args:
            model: PyTorch model
            weights: Flattened weight tensor
        """
        start_idx = 0
        for param in model.parameters():
            end_idx = start_idx + param.numel()
            param.data = weights[start_idx:end_idx].view_as(param.data)
            start_idx = end_idx
            
    def compute_loss(self, model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Compute loss for a model on a data loader.
        
        Args:
            model: PyTorch model
            data_loader: Data loader
            criterion: Loss function
            
        Returns:
            Average loss
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                
        return total_loss / total_samples
        
    def compute_loss_surface(self, x_direction: torch.Tensor, y_direction: torch.Tensor, 
                           x_range: Tuple[float, float] = (-1, 1), 
                           y_range: Tuple[float, float] = (-1, 1), 
                           resolution: int = 51, save_dir: str = './loss_surface') -> str:
        """
        Compute loss surface values on the 2D plane spanned by PCA directions.
        
        Args:
            x_direction: First PCA direction
            y_direction: Second PCA direction
            x_range: Range for x-coordinate (min, max)
            y_range: Range for y-coordinate (min, max)
            resolution: Number of points in each direction
            save_dir: Directory to save surface data
            
        Returns:
            Path to the saved surface file
        """
        os.makedirs(save_dir, exist_ok=True)
        surface_file = os.path.join(save_dir, 'curve_loss_surface.h5')
        
        # Check if surface already exists
        if os.path.exists(surface_file):
            print(f"Loading existing loss surface from {surface_file}")
            return surface_file
            
        print("Computing loss surface using curve PCA components...")
        
        # Create coordinate grids
        x_coords = np.linspace(x_range[0], x_range[1], resolution)
        y_coords = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Initialize loss matrix
        loss_matrix = np.zeros((resolution, resolution))
        
        # Get base model weights (use model A as base)
        if self.model_A is None:
            raise ValueError("Model A not loaded. Call load_models_and_curve() first.")
        base_weights = self.get_model_weights(self.model_A)
        
        # Compute loss at each point
        criterion = nn.CrossEntropyLoss()
        
        for i in range(resolution):
            for j in range(resolution):
                # Compute new weights using PCA directions
                new_weights = base_weights + x_coords[i] * x_direction.to(self.device) + y_coords[j] * y_direction.to(self.device)
                
                # Create temporary model with new weights
                if self.model_A is None:
                    raise ValueError("Model A not loaded. Call load_models_and_curve() first.")
                model_class = type(self.model_A)
                
                # Get model config
                if self.model_name == 'FC':
                    config = {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10}
                elif self.model_name == 'MlpNet':
                    config = {'input_dim': 784, 'num_classes': 10}
                else:
                    raise ValueError(f"Unsupported model: {self.model_name}")
                    
                temp_model = model_class(**config)
                temp_model.to(self.device)
                self.set_model_weights(temp_model, new_weights)
                
                # Compute loss
                if self.test_loader is None:
                    raise ValueError("Test loader not setup. Call setup_data_loaders() first.")
                loss = self.compute_loss(temp_model, self.test_loader, criterion)
                loss_matrix[j, i] = loss
                
                print(f"Progress: {i*resolution + j + 1}/{resolution*resolution} - Loss: {loss:.4f}")
                
        # Save surface data
        with h5py.File(surface_file, 'w') as f:
            f.create_dataset('xcoordinates', data=x_coords)
            f.create_dataset('ycoordinates', data=y_coords)
            f.create_dataset('train_loss', data=loss_matrix)
            f.create_dataset('test_loss', data=loss_matrix)  # Using test loss for both
            
        print(f"Loss surface saved to {surface_file}")
        return surface_file
        
    def plot_2d_contour(self, surface_file: str, save_dir: str = './plots', show: bool = False) -> None:
        """
        Create 2D contour plot of the loss surface.
        
        Args:
            surface_file: Path to the surface data file
            save_dir: Directory to save plots
            show: Whether to display the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with h5py.File(surface_file, 'r') as f:
            x_coords = np.array(f['xcoordinates'])
            y_coords = np.array(f['ycoordinates'])
            loss_matrix = np.array(f['test_loss'])
            
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create contour plot
        plt.figure(figsize=(12, 10))
        
        # Create filled contour plot
        filled_contour = plt.contourf(X, Y, loss_matrix, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(filled_contour, label='Loss')
        
        # Add contour lines
        contour = plt.contour(X, Y, loss_matrix, levels=20, colors='black', alpha=0.6, linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8)
        
        # Add curve trajectory
        if self.model_A is not None and self.model_B is not None and self.curve_mlp is not None:
            self._plot_curve_trajectory(X, Y, x_coords, y_coords)
        
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.title(f'Loss Landscape Contour Plot - {self.model_name} on {self.dataset_name}\nLearned Curve PCA Components')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'curve_loss_contour_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Contour plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def _plot_curve_trajectory(self, X: np.ndarray, Y: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
        """
        Plot the learned curve trajectory on the loss landscape.
        
        Args:
            X: X coordinate meshgrid
            Y: Y coordinate meshgrid
            x_coords: X coordinate values
            y_coords: Y coordinate values
        """
        # Sample points along the curve
        if self.model_A is None or self.model_B is None or self.curve_mlp is None:
            return
            
        flat_A = torch.cat([p.view(-1) for p in self.model_A.parameters()])
        flat_B = torch.cat([p.view(-1) for p in self.model_B.parameters()])
        
        t_values = np.linspace(0, 1, 100)
        curve_points = []
        
        for t in t_values:
            t_tensor = torch.tensor(t, dtype=flat_A.dtype, device=flat_A.device)
            w_curve = self.curve_mlp(t_tensor, flat_A, flat_B).detach()
            
            # Project onto PCA directions
            base_weights = torch.cat([p.view(-1) for p in self.model_A.parameters()])
            diff = w_curve - base_weights
            
            if self.pca_components is not None:
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
        plt.plot(curve_points[50, 0], curve_points[50, 1], 'mo', markersize=8, label='Midpoint (t=0.5)')
        
        plt.legend()
            
    def plot_3d_surface(self, surface_file: str, save_dir: str = './plots', show: bool = False) -> None:
        """
        Create 3D surface plot of the loss landscape.
        
        Args:
            surface_file: Path to the surface data file
            save_dir: Directory to save plots
            show: Whether to display the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with h5py.File(surface_file, 'r') as f:
            x_coords = np.array(f['xcoordinates'])
            y_coords = np.array(f['ycoordinates'])
            loss_matrix = np.array(f['test_loss'])
            
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        surf = ax.plot_surface(X, Y, loss_matrix, cmap='viridis',  # type: ignore
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('1st Principal Component')
        ax.set_ylabel('2nd Principal Component')
        ax.set_zlabel('Loss')  # type: ignore
        ax.set_title(f'3D Loss Landscape - {self.model_name} on {self.dataset_name}\nLearned Curve PCA Components')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'curve_loss_3d_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"3D surface plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_curve_dimensionality(self, save_dir: str = './plots', show: bool = False) -> None:
        """
        Plot the intrinsic dimensionality analysis of the learned curve.
        
        Args:
            save_dir: Directory to save plots
            show: Whether to display the plot
        """
        if self.model_A is None or self.model_B is None or self.curve_mlp is None:
            raise ValueError("Models and curve MLP must be loaded first")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Compute curve dimensionality
        n_components, explained_variance = self.curve_mlp.evaluate_curve_dimensionality(
            self.model_A, self.model_B, num_points=1000, plot=False
        )
        
        # Create plot
        plt.figure(figsize=(10, 8))
        cumulative = np.cumsum(explained_variance)
        
        plt.plot(np.arange(1, len(cumulative)+1), cumulative*100, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=99, color='r', linestyle='--', alpha=0.7, label='99% variance threshold')
        plt.axvline(x=float(n_components), color='g', linestyle='--', alpha=0.7, 
                   label=f'{n_components} components needed')
        
        plt.xlabel('Number of Principal Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance (%)', fontsize=12)
        plt.title(f'Intrinsic Dimensionality of Learned Curve\n{self.model_name} on {self.dataset_name}', fontsize=14)
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
        
        # Save plot
        plot_path = os.path.join(save_dir, f'curve_dimensionality_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Curve dimensionality plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def visualize_curve_loss_landscape(self, model_A_path: str, model_B_path: str, curve_mlp_path: str,
                                     data_path: str = './data', 
                                     x_range: Tuple[float, float] = (-1, 1), 
                                     y_range: Tuple[float, float] = (-1, 1), 
                                     resolution: int = 51,
                                     save_dir: str = './curve_loss_landscape_results', 
                                     show: bool = False) -> None:
        """
        Complete curve-based loss landscape visualization pipeline.
        
        Args:
            model_A_path: Path to model A checkpoint
            model_B_path: Path to model B checkpoint
            curve_mlp_path: Path to the trained curve MLP
            data_path: Path to the dataset
            x_range: Range for x-coordinate
            y_range: Range for y-coordinate
            resolution: Resolution of the loss surface
            save_dir: Directory to save all results
            show: Whether to display plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting curve-based loss landscape visualization...")
        
        # Load models and curve
        self.load_models_and_curve(model_A_path, model_B_path, curve_mlp_path)
        
        # Setup data loaders
        self.setup_data_loaders(data_path)
        
        # Compute PCA components from curve
        pca_dir = os.path.join(save_dir, 'pca_components')
        x_direction, y_direction, explained_variance = self.compute_pca_components_from_curve(num_points=1000, save_dir=pca_dir)
        
        # Compute loss surface
        surface_dir = os.path.join(save_dir, 'loss_surface')
        surface_file = self.compute_loss_surface(x_direction, y_direction, x_range, y_range, resolution, surface_dir)
        
        # Create visualizations
        plots_dir = os.path.join(save_dir, 'plots')
        
        # Curve dimensionality analysis
        self.plot_curve_dimensionality(plots_dir, show)
        
        # 2D contour plot
        self.plot_2d_contour(surface_file, plots_dir, show)
        
        # 3D surface plot
        self.plot_3d_surface(surface_file, plots_dir, show)
        
        print(f"\nCurve-based loss landscape visualization complete!")
        print(f"Results saved in: {save_dir}")
        print(f"Explained variance ratio: {explained_variance}")
        print(f"1st PC explains {explained_variance[0]*100:.1f}% of variance")
        print(f"2nd PC explains {explained_variance[1]*100:.1f}% of variance")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Curve-based Loss Landscape Visualization using PCA')
    
    # Model and dataset parameters
    parser.add_argument('--model_name', type=str, default='MlpNet', 
                       choices=['FC', 'MlpNet'], help='Model architecture')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                       choices=['MNIST', 'CIFAR10'], help='Dataset name')
    
    # Model paths
    parser.add_argument('--model_A_path', type=str, required=True,
                       help='Path to model A checkpoint')
    parser.add_argument('--model_B_path', type=str, required=True,
                       help='Path to model B checkpoint')
    parser.add_argument('--curve_mlp_path', type=str, required=True,
                       help='Path to the trained curve MLP')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for data loading')
    
    # Visualization parameters
    parser.add_argument('--x_range', type=float, nargs=2, default=[-1, 1],
                       help='Range for x-coordinate (min max)')
    parser.add_argument('--y_range', type=float, nargs=2, default=[-1, 1],
                       help='Range for y-coordinate (min max)')
    parser.add_argument('--resolution', type=int, default=51,
                       help='Resolution of the loss surface')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./curve_loss_landscape_results',
                       help='Directory to save results')
    parser.add_argument('--show', action='store_true',
                       help='Display plots (not recommended for headless environments)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use for computations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create visualizer
    visualizer = CurveLossLandscapeVisualizer(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        device=device
    )
    
    # Run visualization
    try:
        visualizer.visualize_curve_loss_landscape(
            model_A_path=args.model_A_path,
            model_B_path=args.model_B_path,
            curve_mlp_path=args.curve_mlp_path,
            data_path=args.data_path,
            x_range=tuple(args.x_range),
            y_range=tuple(args.y_range),
            resolution=args.resolution,
            save_dir=args.save_dir,
            show=args.show
        )
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 