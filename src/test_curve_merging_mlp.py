"""
Trains two models on CIFAR10 dataset and merges them using curve ensembling with MLP.
"""

from enum import Enum
import tyro
from typing import List
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
from curves_MLP import (
    CurveMLP,
    stratified_split,
    split_classes_with_sampled_fraction,
    ExperimentConfig,
    InterpolationType,
    TrainingBufferConfig,
    ModelToMergeConfig,
    CurveConfigMLP
)
from dataclasses import dataclass
import argparse
import wandb


def main():
    # Load experiment configuration
    cfg: ExperimentConfig = tyro.cli(ExperimentConfig)

    run = wandb.init(entity="Continual_Learning-DAL",
                     project="Model Path Fusion - CurveNet MLP Coeffs",
                     config=cfg)

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
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    #80%/20% for A/B for first half of the classes and 80%/20% for B/A for second half of the classes
    train_subset_A, train_subset_B = split_classes_with_sampled_fraction(full_train)

    # Create data loaders for base models
    train_loader_A = DataLoader(
        train_subset_A,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers
    )
    train_loader_B = DataLoader(
        train_subset_B,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=cfg.model.num_workers
    )

    # Create a training buffer for curve fitting (stratified split)
    buffer_subset, _ = stratified_split(full_train, cfg.buffer.percentage)
    training_buffer = DataLoader(
        buffer_subset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers
    )

    # Initialize base models A and B
    model_A = mlpnet.MlpNetBase(
        input_dim=cfg.model.input_dim,
        num_classes=cfg.model.output_dim,
        hidden_dims=cfg.model.hidden_dims
    ).to(device)
    model_B = mlpnet.MlpNetBase(
        input_dim=cfg.model.input_dim,
        num_classes=cfg.model.output_dim,
        hidden_dims=cfg.model.hidden_dims
    ).to(device)

    # Paths for checkpoints
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    path_A = os.path.join(ckpt_dir, "cifar10_model_A.pth")
    path_B = os.path.join(ckpt_dir, "cifar10_model_B.pth")

    # Train or load base models
    #if os.path.exists(path_A) and os.path.exists(path_B):
        #print("Loading pre-trained models...")
        #model_A.load_state_dict(torch.load(path_A))
        #model_B.load_state_dict(torch.load(path_B))
    #else:
    print("Training Model A...")
    train_model(
        cfg.model,
        model_A,
        train_loader_A,
        test_loader,
        epochs=cfg.model.model_epochs,
        device=device
    )
    print("Training Model B...")
    train_model(
        cfg.model,
        model_B,
        train_loader_B,
        test_loader,
        epochs=cfg.model.model_epochs,
        device=device
    )
    torch.save(model_A.state_dict(), path_A)
    torch.save(model_B.state_dict(), path_B)
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
        in_features=W1.numel() * 2,
        out_features=W1.numel(),
        bias=True,
        hidden_dim=cfg.curve.hidden_dim
    ).to(device)

    # Fit CurveMLP with the training buffer
    curve_mlp.fit(
        training_buffer,
        test_loader,
        cfg.curve,
        model_A,
        model_B
    )

    # Save final models
    torch.save(model_A.state_dict(), path_A)
    torch.save(model_B.state_dict(), path_B)
    print("Models and merged outputs saved to checkpoints directory")

if __name__ == "__main__":
    main()

