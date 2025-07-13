#!/usr/bin/env python3
"""
Example script demonstrating how to use the loss landscape visualization.

This script shows how to:
1. Train multiple models for PCA analysis
2. Use the LossLandscapeVisualizer to create visualizations
3. Interpret the results

Usage:
    python example_loss_landscape.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loss_landscape_visualisation import LossLandscapeVisualizer
from models.fcmodel import FCModel
from datasets import HeteroMNIST


def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    """
    Train a model and return training history.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        List of validation accuracies
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    history = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(model, FCModel.base):
                inputs = inputs.view(inputs.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if isinstance(model, FCModel.base):
                    inputs = inputs.view(inputs.size(0), -1)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        history.append(accuracy)
        print(f'Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%')
    
    return history


def create_sample_models(num_models=5, device='cpu'):
    """
    Create and train multiple models for PCA analysis.
    
    Args:
        num_models: Number of models to create
        device: Device to train on
        
    Returns:
        List of trained models
    """
    print("Creating sample models for loss landscape visualization...")
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = HeteroMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = HeteroMNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)  # type: ignore
    
    # Create models with different random seeds
    models = []
    checkpoints_dir = './sample_checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    for i in range(num_models):
        print(f"\nTraining model {i+1}/{num_models}")
        
        # Set different random seed for each model
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Create model
        model = FCModel.base(input_dim=784, hidden_dims=[512, 256], output_dim=10)
        
        # Train model
        history = train_model(model, train_loader, test_loader, epochs=5, device=device)
        
        # Save model
        checkpoint_path = os.path.join(checkpoints_dir, f'model_{i}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': model.get_model_config(),
            'val_acc': history[-1],
            'test_acc': history[-1],
            'epoch': 5
        }, checkpoint_path)
        
        models.append(model)
        print(f"Model {i+1} saved to {checkpoint_path}")
    
    return checkpoints_dir


def main():
    """Main function to demonstrate loss landscape visualization."""
    print("Loss Landscape Visualization Example")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create sample models if they don't exist
    checkpoints_dir = './sample_checkpoints'
    if not os.path.exists(checkpoints_dir) or len(os.listdir(checkpoints_dir)) < 3:
        print("Creating sample models...")
        create_sample_models(num_models=5, device=device)
    else:
        print("Using existing sample models...")
    
    # Create visualizer
    visualizer = LossLandscapeVisualizer(
        model_name='FC',
        dataset_name='MNIST',
        device=device
    )
    
    # Run visualization with lower resolution for faster execution
    print("\nStarting loss landscape visualization...")
    try:
        visualizer.visualize_loss_landscape(
            model_folder=checkpoints_dir,
            data_path='./data',
            x_range=(-0.5, 0.5),  # Smaller range for faster computation
            y_range=(-0.5, 0.5),
            resolution=21,  # Lower resolution for faster computation
            save_dir='./example_results',
            show=False
        )
        
        print("\nVisualization complete! Check the './example_results' directory for:")
        print("- 2D contour plots")
        print("- 3D surface plots") 
        print("- Training trajectory plots")
        print("- PCA directions and loss surface data")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 