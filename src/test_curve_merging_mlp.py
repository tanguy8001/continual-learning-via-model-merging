"""
Trains two models on CIFAR10/MNIST dataset and merges them using curve ensembling with MLP.
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

from data import double_loaders, create_fused_loader

from models import mlpnet, fcmodel
from curve_merging import (
    train_model,
    curve_ensembling,
    CurveConfig,
    evaluate_model,
)
from curves_MLP import CurveMLP

def main():

    config = CurveConfig(
        dataset="MNIST",
        #input_dim=3072,  # 32x32x3
        input_dim=784,
        batch_size=128,
        model_epochs=10,
        epochs=10  # for curve training
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)

    data_loaders, num_classes = double_loaders(
        dataset=config.dataset,
        path=data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform_name=config.transform,
        digit=config.test_digit,
        cifar_class=config.cifar_class,
    )
    
    fused_loader = create_fused_loader(
        data_loaders['trainA'],
        data_loaders['trainB'],
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    model_A = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
    ).to(device)
    model_B = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
    ).to(device)
    
    model_A_path = f"checkpoints/{config.dataset.lower()}_model_A.pth"
    model_B_path = f"checkpoints/{config.dataset.lower()}_model_B.pth"
    
    if os.path.exists(model_A_path) and os.path.exists(model_B_path):
        print("Loading pre-trained models...")
        model_A.load_state_dict(torch.load(model_A_path))
        model_B.load_state_dict(torch.load(model_B_path))
    else:
        print("Training Model A...")
        train_model(config, model_A, data_loaders['trainA'], data_loaders['test'], epochs=config.model_epochs, device=device)
        print("Training Model B...")
        train_model(config, model_B, data_loaders['trainB'], data_loaders['test'], epochs=config.model_epochs, device=device)
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model_A.state_dict(), model_A_path)
        torch.save(model_B.state_dict(), model_B_path)
        print("Models saved to checkpoints directory")
    
    # Evaluate individual models
    acc_A = evaluate_model(model_A, data_loaders['test'], device)
    acc_B = evaluate_model(model_B, data_loaders['test'], device)
    
    print(f"Model A accuracy: {acc_A:.2f}%")
    print(f"Model B accuracy: {acc_B:.2f}%")
    
    # Merge models using CurveMLP
    print("Merging models using CurveMLP...")
    
    W1 = torch.cat([p.view(-1) for p in model_A.parameters()])
    W2 = torch.cat([p.view(-1) for p in model_B.parameters()])
    
    num_params = W1.numel() # Total number of parameters in one model
    print(f"Initializing CurveMLP with num_params: {num_params}, bias: True, hidden_dim: 32")
    curve_mlp = CurveMLP(
        num_params=num_params,
        bias=True,
        hidden_dim=32
    ).to(device)

    # Train the CurveMLP model
    curve_mlp.fit(fused_loader, data_loaders['test'], config, model_A, model_B)


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
    torch.save(model_A.state_dict(), f"checkpoints/{config.dataset.lower()}_model_A.pth")
    torch.save(model_B.state_dict(), f"checkpoints/{config.dataset.lower()}_model_B.pth")
    torch.save(curve_mlp.state_dict(), f"checkpoints/{config.dataset.lower()}_curve_mlp.pth")
    
    print("Models saved to checkpoints directory")

if __name__ == "__main__":
    main()
