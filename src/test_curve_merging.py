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
import copy
from tqdm import tqdm

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
    hidden_dims = config.hidden_dims
    num_classes = 10
    
    
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
        dataset=config.dataset,
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name=config.transform,
        digit=config.test_digit,
        cifar_class=config.cifar_class,
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
        input_dim=config.input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
    )

    model_B = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
    )

    target_model = fcmodel.FCModelBase(
        input_dim=config.input_dim,
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
        model_path="C:/Users/tangu/OneDrive/Documents/ETHZ/Deep Learning/continual-learning-via-model-merging/checkpoints/model_A",
        device=device,
        epochs=model_epochs
    )

    print("\nTraining Model B...")
    history['model_B'] = train_and_evaluate_model(
        model=model_B,
        train_loader=data_loaders['trainB'],
        test_loader=data_loaders['test'],
        config=config,
        model_path="C:/Users/tangu/OneDrive/Documents/ETHZ/Deep Learning/continual-learning-via-model-merging/checkpoints/model_B",
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
        input_dim=config.input_dim,
    )
    
    merged_model = curve_ensembling(
        config=config,
        models=[model_A, model_B],
        target_model=target_model,
        train_loader=fused_loader,  # Use fused loader for curve training
        test_loader=data_loaders['test'],
        device=device,
        num_classes=num_classes,
        input_dim=config.input_dim
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

class ODEFunc(nn.Module):
    def __init__(self, param_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim + 1, 256),  # +1 for time
            nn.Tanh(),
            nn.Linear(256, param_dim)
        )
        
    def forward(self, t, y):
        """t: time, y: parameters"""
        # Ensure t is properly shaped for concatenation with y
        if not torch.is_tensor(t):
            t = torch.tensor(t).float()
        
        # Convert t to a 1D tensor with a single element
        t = t.reshape(-1)
        
        # Make sure y is a 1D tensor
        y_flat = y.reshape(-1)
        
        # Add t as an extra element to the parameters
        # Use unsqueeze to add a batch dimension for the network
        input_tensor = torch.cat([t, y_flat]).unsqueeze(0)
        
        # Forward through network and remove batch dimension
        return self.net(input_tensor).squeeze(0)

def compute_ode_loss(ode_func, model_template, theta_0, theta_1, dataloader, n_samples=10):
    # Use torchdiffeq for ODE solving
    from torchdiffeq import odeint
    
    # Sample timepoints
    t_samples = torch.linspace(0, 1, n_samples)
    
    # Solve ODE
    ode_solution = odeint(ode_func, theta_0, t_samples)
    
    # Compute loss
    acc_loss = 0.0
    endpoint_loss = torch.norm(ode_solution[0] - theta_0)**2 + torch.norm(ode_solution[-1] - theta_1)**2
    
    for i, theta_t in enumerate(ode_solution):
        # Create model with these parameters
        model = copy.deepcopy(model_template)
        nn.utils.vector_to_parameters(theta_t, model.parameters())
        
        # Compute accuracy on batch
        for x, y in dataloader:
            output = model(x)
            acc_loss += F.cross_entropy(output, y)
    
    # Dynamics regularization
    reg_loss = 0.0
    for t in torch.linspace(0, 1, 10):
        t_tensor = torch.tensor([t])
        theta_t = nn.utils.parameters_to_vector(model.parameters())
        dtheta_dt = ode_func(t_tensor, theta_t)
        reg_loss += torch.norm(dtheta_dt)**2
    
    return acc_loss + 0.1 * endpoint_loss + 0.01 * reg_loss

def test_ode_merging():
    """Test Neural ODE approach for model merging."""
    # Setup similar to test_curve_merging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use the same configuration as in test_curve_merging
    config = CurveConfig(
        learning_rate=0.05,
    )
    
    # Load data in the same way
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)
    
    data_loaders, num_classes = double_loaders(
        dataset=config.dataset,
        path=data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform_name="MLPNET",
        digit=config.test_digit,
        cifar_class=config.cifar_class,
    )
    
    # Create fused loader for training
    fused_loader = create_fused_loader(
        data_loaders['trainA'],
        data_loaders['trainB'],
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # Create models - same as in test_curve_merging
    model_A = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    ).to(device)
    
    model_B = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    ).to(device)
    
    # Load pre-trained models (optional)
    # If you already have trained models, load them here
    # Otherwise, train them as in test_curve_merging
    
    # Extract parameters as vectors
    theta_0 = nn.utils.parameters_to_vector(model_A.parameters()).detach()
    theta_1 = nn.utils.parameters_to_vector(model_B.parameters()).detach()
    
    # Initialize ODE function
    param_dim = theta_0.numel()
    ode_func = ODEFunc(param_dim).to(device)
    
    # Initialize target model for ODE approach
    target_model_ode = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    ).to(device)
    
    # Train ODE function
    from torchdiffeq import odeint
    
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.01)
    n_epochs = 10
    
    # Epoch progress bar
    for epoch in tqdm(range(n_epochs), desc="Training ODE function"):
        ode_func.train()
        
        # Train for one epoch
        epoch_loss = 0.0
        batch_count = 0
        
        # Batch progress bar
        train_bar = tqdm(fused_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Sample random time points
            t_samples = torch.linspace(0, 1, 5).to(device)
            
            # Solve ODE
            theta_samples = odeint(ode_func, theta_0, t_samples)

            print("theta_samples: ", theta_samples)
            
            # Compute loss at each sampled point
            total_loss = 0
            for i, theta_t in enumerate(theta_samples):
                # Create model with these parameters
                nn.utils.vector_to_parameters(theta_t, target_model_ode.parameters())
                
                # Forward pass
                outputs = target_model_ode(x)
                loss = F.cross_entropy(outputs, y)
                
                # Add to total loss
                total_loss += loss
            
            # Add endpoint constraints
            endpoint_loss = torch.norm(theta_samples[0] - theta_0)**2 + torch.norm(theta_samples[-1] - theta_1)**2
            
            # Add regularization to encourage smooth paths
            reg_loss = 0
            for t in torch.linspace(0, 1, 10).to(device):
                t_tensor = t.view(1).to(device)
                dtheta_dt = ode_func(t_tensor, theta_0.clone())
                reg_loss += torch.norm(dtheta_dt)**2
            
            # Combined loss
            loss = total_loss + 0.1 * endpoint_loss + 0.01 * reg_loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            batch_count += 1
            train_bar.set_postfix({"Loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    # Find minimum loss point on ODE path
    ode_func.eval()
    with torch.no_grad():
        t_grid = torch.linspace(0, 1, 21).to(device)
        theta_grid = odeint(ode_func, theta_0, t_grid)
        
        min_loss = float('inf')
        best_t = 0.5
        best_theta = None
        
        # Add tqdm progress bar for evaluation
        eval_bar = tqdm(enumerate(t_grid), total=len(t_grid), desc="Finding best point on curve")
        for i, t in eval_bar:
            theta_t = theta_grid[i]
            nn.utils.vector_to_parameters(theta_t, target_model_ode.parameters())
            
            # Evaluate loss on validation set
            total_loss = 0
            for x, y in data_loaders['test']:
                x, y = x.to(device), y.to(device)
                outputs = target_model_ode(x)
                loss = F.cross_entropy(outputs, y)
                total_loss += loss.item()
            
            if total_loss < min_loss:
                min_loss = total_loss
                best_t = t.item()
                best_theta = theta_t.clone()
                
            eval_bar.set_postfix({"t": t.item(), "loss": total_loss, "best_t": best_t})
        
        print(f"Best ODE point: t={best_t:.3f}, Loss={min_loss:.4f}")
        
        # Set target model to best parameters
        nn.utils.vector_to_parameters(best_theta, target_model_ode.parameters())
    
    # Evaluate ODE model
    ode_acc = evaluate_model(target_model_ode, data_loaders['test'], device)
    print(f"\nODE Model Accuracy: {ode_acc:.2f}%")
    
    # Now run the standard Bezier curve approach for comparison
    target_model_bezier = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim
    ).to(device)
    
    bezier_model = curve_ensembling(
        config=config,
        models=[model_A, model_B],
        target_model=target_model_bezier,
        train_loader=fused_loader,
        test_loader=data_loaders['test'],
        device=device,
        num_classes=config.output_dim,
        input_dim=config.input_dim
    )
    
    # Evaluate Bezier model
    bezier_acc = evaluate_model(bezier_model, data_loaders['test'], device)
    print(f"\nBezier Model Accuracy: {bezier_acc:.2f}%")
    
    # Compare results
    print("\nComparison:")
    print(f"ODE Model: {ode_acc:.2f}%")
    print(f"Bezier Model: {bezier_acc:.2f}%")
    
    return {
        'ode_acc': ode_acc,
        'bezier_acc': bezier_acc,
        'model_A': model_A,
        'model_B': model_B,
        'ode_model': target_model_ode,
        'bezier_model': bezier_model,
        'best_t': best_t
    }

if __name__ == "__main__":
    # Run the original Bezier curve test
    print("\n=== Testing Bezier Curve Merging ===")
    #history, bezier_model = test_curve_merging()
    
    # Run the Neural ODE test
    print("\n=== Testing Neural ODE Merging ===")
    ode_results = test_ode_merging()
    
    # Final comparison
    print("\n=== Final Comparison ===")
    #print(f"Bezier Curve Accuracy: {history['merged'][-1]:.2f}%")
    print(f"Neural ODE Accuracy: {ode_results['ode_acc']:.2f}%")
    print(f"Neural ODE Best t-value: {ode_results['best_t']:.3f}")