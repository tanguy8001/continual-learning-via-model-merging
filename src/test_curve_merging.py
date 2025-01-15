"""
Trains two models on a heterogeneous dataset split (MNIST or CIFAR10).
Then merges the trained models using curve ensembling.
Saves the checkpoint for the computed minimum-loss curve via the curve_ensembling function.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import double_loaders
from models import mlpnet, fcmodel
from curve_merging import (
    train_model,
    curve_ensembling,
    CurveConfig
)

class MNIST:
    """MNIST-specific transforms"""
    def __init__(self):
        self.train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test = self.train

class Transforms:
    """Container for dataset-specific transforms."""
    MNIST = MNIST()

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def create_fused_loader(loader_A, loader_B, batch_size, num_workers):
    """Create a fused data loader from two separate loaders."""
    # Get underlying datasets
    dataset_A = loader_A.dataset
    dataset_B = loader_B.dataset
    
    # Combine datasets
    fused_dataset = ConcatDataset([dataset_A, dataset_B])
    
    # Create new loader with combined data
    fused_loader = DataLoader(
        fused_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return fused_loader

def test_curve_merging():
    """Main test function for curve merging."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_epochs = 20
    #config = CurveConfig( 
    #    learning_rate=0.007
    #)
    config = CurveConfig( 
        learning_rate=0.05
    )

    # Parameters
    batch_size = 128
    num_workers = 2
    #input_dim = 28 * 28  # MNIST image size
    input_dim = 32 * 32 * 3
    hidden_dims = config.hidden_dims
    num_classes = 10
    test_digit = "dog"  # The digit to split on
    cifar_class = "dog"
    transform_name = "MLPNET"
    dataset = "CIFAR10"
    
    
    # Create data directory if it doesn't exist
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)
    
    #data_loaders, num_classes = double_loaders(
    #    dataset="MNIST",
    #    path=data_path,
    #    batch_size=batch_size,
    #    num_workers=num_workers,
    #    transform_name="MLPNET",
    #    digit=test_digit
    #)

    data_loaders, num_classes = double_loaders(
        dataset="CIFAR10",
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name="MLPNET",
        digit=test_digit,
        cifar_class=cifar_class,
    )
    
    # Create fused loader for curve training
    fused_loader = create_fused_loader(
        data_loaders['trainA'],
        data_loaders['trainB'],
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create models
    #model_A = mlpnet.MlpNetBase(input_dim=input_dim, num_classes=num_classes).to(device)
    #model_B = mlpnet.MlpNetBase(input_dim=input_dim, num_classes=num_classes).to(device)
    model_A = fcmodel.FCModelBase(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
    )

    model_B = fcmodel.FCModelBase(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
    )

    target_model = fcmodel.FCModelBase(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes
    )
    
    # Training history
    history = {
        'model_A': [],
        'model_B': [],
        'merged': []
    }


    print("\nTraining Model A...")
    history['model_A'] = train_and_evaluate_model(
        model=model_A,
        train_loader=data_loaders['trainA'],
        test_loader=data_loaders['test'],
        config=config,
        model_path="/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_A",
        device=device,
        epochs=model_epochs
    )

    print("\nTraining Model B...")
    history['model_B'] = train_and_evaluate_model(
        model=model_B,
        train_loader=data_loaders['trainB'],
        test_loader=data_loaders['test'],
        config=config,
        model_path="/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints/model_B",
        device=device,
        epochs=model_epochs
    )
    
    # Merge models
    print("\nMerging models...")
    config = CurveConfig(
        epochs=10,
        learning_rate=0.07,
        num_bends=3,
        curve="Bezier",
        input_dim=32*32*3,
    )
    
    merged_model = curve_ensembling(
        config=config,
        models=[model_A, model_B],
        target_model=target_model,
        train_loader=fused_loader,  # Use fused loader for curve training
        test_loader=data_loaders['test'],
        device=device,
        num_classes=num_classes,
        input_dim=input_dim
    ).to(device)
    
    # Evaluate merged model
    merged_acc = evaluate_model(merged_model, data_loaders['test'], device)
    history['merged'].append(merged_acc)
    print(f"\nMerged Model Accuracy: {merged_acc:.2f}%")
    
    ## Plot results
    #plt.figure(figsize=(12, 6))
    #plt.plot(history['model_A'], label='Model A')
    #plt.plot(history['model_B'], label='Model B')
    #plt.axhline(y=merged_acc, color='r', linestyle='--', label='Merged Model')
    #plt.title('Model Performance During Training and After Merging')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy (%)')
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    
    # Detailed evaluation
    print("\nDetailed Evaluation:")
    print(f"Model A final accuracy: {history['model_A'][-1]:.2f}%")
    print(f"Model B final accuracy: {history['model_B'][-1]:.2f}%")
    print(f"Merged model accuracy: {merged_acc:.2f}%")
    
    return history, merged_model


def train_and_evaluate_model(model, train_loader, test_loader, config, model_path, device, epochs):
    """
    Trains a model, evaluates it, and saves the best and final models.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing/validation data.
        config: Configuration dictionary for training.
        model_path: Directory to save the model checkpoints.
        device: The device (CPU/GPU) for training.
        epochs: Number of epochs to train.
    """
    os.makedirs(model_path, exist_ok=True)
    best_val_acc = 0
    history = []

    print(model)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # Train the model for one epoch
        train_model(
            config,
            model,
            train_loader,
            test_loader,
            epochs=1,
            device=device,
            learning_rate=config.learning_rate
        )
        
        # Evaluate the model
        val_acc = evaluate_model(model, test_loader, device)
        history.append(val_acc)
        print(f"Epoch {epoch}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(model_path, 'best_val_acc_model.pth')
            save_model(model, config, epoch, val_acc, save_path)
            print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")
    
    # Save the final model
    final_save_path = os.path.join(model_path, 'final_model.pth')
    save_model(model, config, epochs, val_acc, final_save_path)
    print(f"Final model saved after {epochs} epochs.")
    
    return history

def save_model(model, config, epoch, val_acc, save_path):
    """
    Saves the model checkpoint.

    Args:
        model: The model to save.
        config: Configuration dictionary.
        epoch: Current epoch number.
        val_acc: Validation accuracy at the time of saving.
        save_path: Path to save the model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'test_acc': val_acc,
        'model_state_dict': model.state_dict(),
        'config': model.get_model_config() 
    }, save_path)

if __name__ == "__main__":
    history, merged_model = test_curve_merging()