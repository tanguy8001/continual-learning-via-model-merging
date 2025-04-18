"""
Trains two models on CIFAR10 dataset and merges them using curve ensembling with MLP.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from models import mlpnet, fcmodel
from curve_merging import (
    train_model,
    curve_ensembling,
    CurveConfig,
    evaluate_model,
    train_model
)
from curves_MLP import CurveMLP

     

def main():
    # Configuration
    config = CurveConfig()
    config.dataset = "CIFAR10"
    config.input_dim = 3072  # 32x32x3
    config.batch_size = 128
    config.model_epochs = 10
    config.epochs = 10  # for curve training
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split training data for two models
    train_size = len(train_dataset)
    indices = list(range(train_size))
    np.random.shuffle(indices)
    
    # Create two different splits of the data
    split_idx = train_size // 2
    indices_A = indices[:split_idx]
    indices_B = indices[split_idx:]
    
    # Create subsets
    train_subset_A = Subset(train_dataset, indices_A)
    train_subset_B = Subset(train_dataset, indices_B)
    
    # Create data loaders
    train_loader_A = DataLoader(
        train_subset_A, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    train_loader_B = DataLoader(
        train_subset_B, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    # Create a combined loader for curve training
    combined_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    # Initialize models
    model_A = mlpnet.MlpNetBase(
        input_dim=config.input_dim,
        num_classes=config.output_dim,
        hidden_dims=config.hidden_dims
    ).to(device)
    
    model_B = mlpnet.MlpNetBase(
        input_dim=config.input_dim,
        num_classes=config.output_dim,
        hidden_dims=config.hidden_dims
    ).to(device)
    
    # Check if models exist and load them if they do
    model_A_path = "checkpoints/cifar10_model_A.pth"
    model_B_path = "checkpoints/cifar10_model_B.pth"
    
    if os.path.exists(model_A_path) and os.path.exists(model_B_path):
        print("Loading pre-trained models...")
        model_A.load_state_dict(torch.load(model_A_path))
        model_B.load_state_dict(torch.load(model_B_path))
    else:
        # Train models on different data splits
        print("Training Model A...")
        train_model(config, model_A, train_loader_A, test_loader, epochs=config.model_epochs, device=device)
        
        print("Training Model B...")
        train_model(config, model_B, train_loader_B, test_loader, epochs=config.model_epochs, device=device)
        
        # Save models after training
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model_A.state_dict(), model_A_path)
        torch.save(model_B.state_dict(), model_B_path)
        print("Models saved to checkpoints directory")
    
    # Evaluate individual models
    acc_A = evaluate_model(model_A, test_loader, device)
    acc_B = evaluate_model(model_B, test_loader, device)
    
    print(f"Model A accuracy: {acc_A:.2f}%")
    print(f"Model B accuracy: {acc_B:.2f}%")
    
    # Merge models using CurveMLP
    print("Merging models using CurveMLP...")
    
    W1 = torch.cat([p.view(-1) for p in model_A.parameters()])
    W2 = torch.cat([p.view(-1) for p in model_B.parameters()])
    
    curve_mlp = CurveMLP(
        in_features=2*W1.numel(),  # t parameter dimension
        out_features=W1.numel(),  # same as total number of parameters
        bias=True,
        hidden_dim=32
    ).to(device)

    # Train the CurveMLP model
    curve_mlp.fit(combined_loader, test_loader,  config, model_A, model_B)


    ## Choose t=0.5 for equal contribution from both models
    #t = torch.tensor([0.5], device=device)
    
    ## Get merged weights
    #merged_weights = curve_mlp(t, W1, W2)
    
    ## Create a new model with merged weights
    #merged_model = mlpnet.MlpNetBase(
        #input_dim=config.input_dim,
        #num_classes=config.output_dim,
        #hidden_dims=config.hidden_dims
    #).to(device)
    
    ## Load merged weights into thenew model
    #offset = 0
    #for param in merged_model.parameters():
        #param_size = param.numel()
        #param.data.copy_(merged_weights[offset:offset + param_size].view(param.size()))
        #offset += param_size
    
    ## Evaluate merged model
    #acc_merged = evaluate_model(merged_model, test_loader, device)
    #print(f"Merged model accuracy: {acc_merged:.2f}%")
    
    # Save models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model_A.state_dict(), "checkpoints/cifar10_model_A.pth")
    torch.save(model_B.state_dict(), "checkpoints/cifar10_model_B.pth")
    #torch.save(merged_model.state_dict(), "checkpoints/cifar10_merged_model.pth")
    
    print("Models saved to checkpoints directory")

if __name__ == "__main__":
    main()

