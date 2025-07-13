#!/usr/bin/env python3
"""
Loss Landscape Visualization using PCA Principal Components

This script visualizes the loss landscape by:
1. Computing PCA directions from model checkpoints
2. Projecting the loss surface onto the 2D plane spanned by the first two principal components
3. Creating contour plots, 3D surfaces, and trajectory visualizations

Usage:
    python loss_landscape_visualisation.py --model_folder checkpoints/ --model_name FC --dataset_name MNIST
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
from torchvision import transforms
import warnings
from typing import List, Tuple, Optional, Any, Union
warnings.filterwarnings('ignore')

# Add the src directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fcmodel import FCModel
from models.mlpnet import MlpNetBase
from datasets import HeteroMNIST
import data


class LossLandscapeVisualizer:
    """
    A class for visualizing loss landscapes using PCA principal components.
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
        self.model: Optional[nn.Module] = None
        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model_name == 'FC':
            config = checkpoint.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
            self.model = FCModel.base(**config)
        elif self.model_name == 'MlpNet':
            config = checkpoint.get('config', {'input_dim': 784, 'num_classes': 10})
            self.model = MlpNetBase(**config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model accuracy: {checkpoint.get('val_acc', 'N/A')}%")
        
    def setup_data_loaders(self, data_path: str = './data', batch_size: int = 128) -> None:
        """
        Setup data loaders for the specified dataset.
        
        Args:
            data_path: Path to the dataset
            batch_size: Batch size for data loading
        """
        if self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = HeteroMNIST(
                root=data_path, 
                train=True, 
                transform=transform,
                download=True
            )
            test_dataset = HeteroMNIST(
                root=data_path, 
                train=False, 
                transform=transform,
                download=True
            )
            
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        # Cast to Dataset type to satisfy type checker
        self.train_loader = DataLoader(
            train_dataset,  # type: ignore
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        self.test_loader = DataLoader(
            test_dataset,  # type: ignore
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Setup data loaders for {self.dataset_name}")
        
    def get_model_weights(self, model: nn.Module) -> torch.Tensor:
        """
        Extract all weights from a model as a flattened tensor.
        
        Args:
            model: PyTorch model
            
        Returns:
            Flattened tensor of all weights
        """
        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.view(-1))
        return torch.cat(weights)
        
    def set_model_weights(self, model: nn.Module, weights: torch.Tensor) -> None:
        """
        Set model weights from a flattened tensor.
        
        Args:
            model: PyTorch model
            weights: Flattened tensor of weights
        """
        start_idx = 0
        for param in model.parameters():
            if param.requires_grad:
                param_size = param.numel()
                param.data = weights[start_idx:start_idx + param_size].view(param.size())
                start_idx += param_size
                
    def compute_loss(self, model: nn.Module, data_loader: DataLoader, criterion: Optional[nn.Module] = None) -> float:
        """
        Compute loss on the given data loader.
        
        Args:
            model: PyTorch model
            data_loader: DataLoader for evaluation
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Average loss value
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if self.model_name == 'FC':
                    inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def compute_pca_directions(self, model_paths: List[str], save_dir: str = './pca_directions') -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Compute PCA directions from multiple model checkpoints.
        
        Args:
            model_paths: List of paths to model checkpoints
            save_dir: Directory to save PCA directions
            
        Returns:
            Tuple of (x_direction, y_direction, explained_variance_ratio)
        """
        os.makedirs(save_dir, exist_ok=True)
        dir_file = os.path.join(save_dir, 'pca_directions.h5')
        
        # Check if PCA directions already exist
        if os.path.exists(dir_file):
            print(f"Loading existing PCA directions from {dir_file}")
            with h5py.File(dir_file, 'r') as f:
                x_direction = torch.from_numpy(np.array(f['xdirection'])).float()
                y_direction = torch.from_numpy(np.array(f['ydirection'])).float()
                explained_variance_ratio = np.array(f['explained_variance_ratio_'])
            return x_direction, y_direction, explained_variance_ratio
        
        print("Computing PCA directions...")
        
        # Load models and extract weights
        weight_vectors = []
        base_model: Optional[nn.Module] = None
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Warning: Model path {model_path} does not exist, skipping...")
                continue
                
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if self.model_name == 'FC':
                config = checkpoint.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
                model = FCModel.base(**config)
            elif self.model_name == 'MlpNet':
                config = checkpoint.get('config', {'input_dim': 784, 'num_classes': 10})
                model = MlpNetBase(**config)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            if base_model is None:
                base_model = model
                
            weights = self.get_model_weights(model)
            weight_vectors.append(weights.cpu().numpy())
            
        if len(weight_vectors) < 2:
            raise ValueError("Need at least 2 model checkpoints for PCA")
            
        # Compute PCA
        weight_matrix = np.array(weight_vectors)
        pca = PCA(n_components=2)
        pca.fit(weight_matrix)
        
        # Get principal components
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        
        # Convert to tensor format for saving
        x_direction = torch.from_numpy(pc1).float()
        y_direction = torch.from_numpy(pc2).float()
        
        # Save PCA directions
        with h5py.File(dir_file, 'w') as f:
            f.create_dataset('xdirection', data=x_direction.numpy())
            f.create_dataset('ydirection', data=y_direction.numpy())
            f.create_dataset('explained_variance_ratio_', data=pca.explained_variance_ratio_)
            f.create_dataset('singular_values_', data=pca.singular_values_)
            f.create_dataset('explained_variance_', data=pca.explained_variance_)
            
        print(f"PCA directions saved to {dir_file}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        return x_direction, y_direction, pca.explained_variance_ratio_
        
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
        surface_file = os.path.join(save_dir, 'loss_surface.h5')
        
        # Check if surface already exists
        if os.path.exists(surface_file):
            print(f"Loading existing loss surface from {surface_file}")
            return surface_file
            
        print("Computing loss surface...")
        
        # Create coordinate grids
        x_coords = np.linspace(x_range[0], x_range[1], resolution)
        y_coords = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Initialize loss matrix
        loss_matrix = np.zeros((resolution, resolution))
        
        # Get base model weights
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        base_weights = self.get_model_weights(self.model)
        
        # Compute loss at each point
        criterion = nn.CrossEntropyLoss()
        
        for i in range(resolution):
            for j in range(resolution):
                # Compute new weights
                new_weights = base_weights + x_coords[i] * x_direction.to(self.device) + y_coords[j] * y_direction.to(self.device)
                
                # Create temporary model with new weights
                if self.model is None:
                    raise ValueError("Model not loaded. Call load_model() first.")
                model_class = type(self.model)
                temp_model = model_class(**self.model.get_model_config())
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
        plt.figure(figsize=(10, 8))
        contour = plt.contour(X, Y, loss_matrix, levels=20, colors='black', alpha=0.6)
        plt.clabel(contour, inline=True, fontsize=8)
        
        # Create filled contour plot
        filled_contour = plt.contourf(X, Y, loss_matrix, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(filled_contour, label='Loss')
        
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.title(f'Loss Landscape Contour Plot - {self.model_name} on {self.dataset_name}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'loss_contour_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Contour plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
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
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        surf = ax.plot_surface(X, Y, loss_matrix, cmap='viridis', 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('1st Principal Component')
        ax.set_ylabel('2nd Principal Component')
        ax.set_zlabel('Loss')  # type: ignore
        ax.set_title(f'3D Loss Landscape - {self.model_name} on {self.dataset_name}')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'loss_3d_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"3D surface plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_trajectory(self, model_paths: List[str], x_direction: torch.Tensor, y_direction: torch.Tensor, 
                       save_dir: str = './plots', show: bool = False) -> None:
        """
        Plot training trajectory on the loss landscape.
        
        Args:
            model_paths: List of model checkpoint paths
            x_direction: First PCA direction
            y_direction: Second PCA direction
            save_dir: Directory to save plots
            show: Whether to display the plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load base model weights (first checkpoint)
        base_checkpoint = torch.load(model_paths[0], map_location=self.device)
        if self.model_name == 'FC':
            config = base_checkpoint.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
            base_model = FCModel.base(**config)
        elif self.model_name == 'MlpNet':
            config = base_checkpoint.get('config', {'input_dim': 784, 'num_classes': 10})
            base_model = MlpNetBase(**config)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        base_model.load_state_dict(base_checkpoint['model_state_dict'])
        base_weights = self.get_model_weights(base_model)
        
        # Project each checkpoint onto PCA directions
        x_coords = []
        y_coords = []
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                continue
                
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if self.model_name == 'FC':
                config = checkpoint.get('config', {'input_dim': 784, 'hidden_dims': [512, 256], 'output_dim': 10})
                model = FCModel.base(**config)
            elif self.model_name == 'MlpNet':
                config = checkpoint.get('config', {'input_dim': 784, 'num_classes': 10})
                model = MlpNetBase(**config)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            weights = self.get_model_weights(model)
            diff = weights - base_weights
            
            # Project onto PCA directions
            x_coord = torch.dot(diff, x_direction.to(self.device)).item()
            y_coord = torch.dot(diff, y_direction.to(self.device)).item()
            
            x_coords.append(x_coord)
            y_coords.append(y_coord)
            
        # Plot trajectory
        plt.figure(figsize=(10, 8))
        plt.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=6, label='Training Trajectory')
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
        
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.title(f'Training Trajectory - {self.model_name} on {self.dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(save_dir, f'trajectory_{self.model_name}_{self.dataset_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def visualize_loss_landscape(self, model_folder: str, data_path: str = './data', 
                               x_range: Tuple[float, float] = (-1, 1), 
                               y_range: Tuple[float, float] = (-1, 1), 
                               resolution: int = 51,
                               save_dir: str = './loss_landscape_results', show: bool = False) -> None:
        """
        Complete loss landscape visualization pipeline.
        
        Args:
            model_folder: Folder containing model checkpoints
            data_path: Path to the dataset
            x_range: Range for x-coordinate
            y_range: Range for y-coordinate
            resolution: Resolution of the loss surface
            save_dir: Directory to save all results
            show: Whether to display plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Find model checkpoints
        model_paths = []
        for file in os.listdir(model_folder):
            if file.endswith('.pth') or file.endswith('.checkpoint'):
                model_paths.append(os.path.join(model_folder, file))
                
        if len(model_paths) < 2:
            raise ValueError(f"Need at least 2 model checkpoints in {model_folder}")
            
        model_paths.sort()  # Sort by filename
        print(f"Found {len(model_paths)} model checkpoints")
        
        # Load the first model as base model
        self.load_model(model_paths[0])
        
        # Setup data loaders
        self.setup_data_loaders(data_path)
        
        # Compute PCA directions
        pca_dir = os.path.join(save_dir, 'pca_directions')
        x_direction, y_direction, explained_variance = self.compute_pca_directions(model_paths, pca_dir)
        
        # Compute loss surface
        surface_dir = os.path.join(save_dir, 'loss_surface')
        surface_file = self.compute_loss_surface(x_direction, y_direction, x_range, y_range, resolution, surface_dir)
        
        # Create visualizations
        plots_dir = os.path.join(save_dir, 'plots')
        
        # 2D contour plot
        self.plot_2d_contour(surface_file, plots_dir, show)
        
        # 3D surface plot
        self.plot_3d_surface(surface_file, plots_dir, show)
        
        # Training trajectory
        self.plot_trajectory(model_paths, x_direction, y_direction, plots_dir, show)
        
        print(f"\nLoss landscape visualization complete!")
        print(f"Results saved in: {save_dir}")
        print(f"Explained variance ratio: {explained_variance}")
        print(f"1st PC explains {explained_variance[0]*100:.1f}% of variance")
        print(f"2nd PC explains {explained_variance[1]*100:.1f}% of variance")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Loss Landscape Visualization using PCA')
    
    # Model and dataset parameters
    parser.add_argument('--model_name', type=str, default='FC', 
                       choices=['FC', 'MlpNet'], help='Model architecture')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                       choices=['MNIST'], help='Dataset name')
    parser.add_argument('--model_folder', type=str, required=True,
                       help='Folder containing model checkpoints')
    
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
    parser.add_argument('--save_dir', type=str, default='./loss_landscape_results',
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
    visualizer = LossLandscapeVisualizer(args.model_name, args.dataset_name, device)
    
    # Run visualization
    try:
        visualizer.visualize_loss_landscape(
            model_folder=args.model_folder,
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
        sys.exit(1)


if __name__ == '__main__':
    main()
