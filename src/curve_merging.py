from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as mods
from models import mlpnet, fcmodel
import curves
from CL.Eval import evaluate_task_accuracies
from CL.Data import get_task_data_with_labels
import logging

def train_model(config, model, train_loader, test_loader, epochs, device, learning_rate=None):
    """Generic training function for a model."""
    model.train()
    lr = learning_rate if learning_rate is not None else config['learning_rate']
    momentum = config['optimizer_momentum']
    weight_decay = config['optimizer_weight_decay']

    optimizer_type = config['optimizer'].upper()

    print(f"Optimizer: {optimizer_type}, LR: {lr}, Momentum: {momentum}, Weight Decay: {weight_decay}")

    if optimizer_type == 'ADAM':
         optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})

        scheduler.step()

        if test_loader:
            test_acc = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.2f}%")
            model.train()

    print('Finished Training')

def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def copy_model_mlpnet(model: nn.Module) -> nn.Module:
    """Create a deep copy of model weights."""
    copy = mlpnet.MlpNet.base()
    copy.load_state_dict({
        name: param.clone() 
        for name, param in model.state_dict().items()
    })
    return copy


def curve_ensembling(
    config: dict,
    models: List[nn.Module],
    target_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    num_classes: int,
    input_dim: int
) -> nn.Module:
    """
    Perform curve ensembling to merge multiple models. 
    Steps for merging the two models are:
        1. Train the curve connecting the two models.
        2. Create the fusion model from it
    
    Args:
        config: Configuration for curve merging
        models: List of models to merge
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to use for computation
        num_classes: Number of output classes
        input_dim: Input dimension
        
    Returns:
        Merged model
    """
    model_name = config['model']
    architecture_builder = getattr(mods, model_name)

    arch_kwargs = {
        'input_dim': input_dim,
        'hidden_dims': config['hidden_dims'],
        'output_dim': num_classes
    }

    curve_type_name = config['curve']
    curve = getattr(curves, curve_type_name)
    
    curve_model = curves.CurveNet(
        num_classes,
        curve,
        architecture_builder.curve,
        config['bezier_num_bends'],
        config['bezier_fix_start'],
        config['bezier_fix_end'],
        architecture_kwargs=arch_kwargs,
    ).to(device)

    curve_model.import_base_parameters(models[0], 0)
    curve_model.import_base_parameters(models[1], config['bezier_num_bends'] - 1)
    
    # Train the curve model
    train_model(
        config=config,
        model=curve_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config['bezier_epochs'],
        learning_rate=config['bezier_lr'],
        device=device
    )
    curve_ckpt_path = config.get('curve_checkpoint_path')
    if curve_ckpt_path:
        curve_net_config = {
            'input_dim': input_dim,
            'hidden_dims': config['hidden_dims'],
            'output_dim': num_classes,
            'num_bends': config['bezier_num_bends'],
            'curve_type': config['curve'],
            'base_model': config['model'],
            'fix_start': config['bezier_fix_start'],
            'fix_end': config['bezier_fix_end']
        }
        checkpoint = {
            'model_state_dict': curve_model.state_dict(),
            'config': curve_net_config
        }
        os.makedirs(os.path.dirname(curve_ckpt_path), exist_ok=True)
        torch.save(checkpoint, curve_ckpt_path)
        print(f"CurveNet checkpoint saved to {curve_ckpt_path}")
    
    # Sample weights from middle of curve
    num_points_sampling = config['bezier_num_points']
    steps = np.linspace(0.0, 1.0, num_points_sampling)
    middle_step = steps[len(steps) // 2]
    fusion_weights = curve_model.weights(torch.tensor([middle_step]))
    
    # Update merged model parameters
    offset = 0
    for parameter in target_model.parameters():
        size = np.prod(parameter.size())
        value_tensor = fusion_weights[offset:offset + size].reshape(parameter.size())
        parameter.data.copy_(torch.from_numpy(value_tensor))
        offset += size

    return target_model


def train_merging_curve(
    seq_data,
    model: nn.Module,
    device: str = "cpu",
    num_classes: int = 10,
    input_dim: int = 784,
    config: Optional[dict] = None
) -> Tuple[nn.Module, List[float]]:
    """
    Train sequential model with curve merging for continual learning.
    
    Args:
        seq_data: Sequential data provider
        model: Initial model
        device: Device to use for computation
        num_classes: Number of output classes
        input_dim: Input dimension
        config: Configuration for curve merging
        
    Returns:
        Tuple of (trained model, task accuracies)
    """
    print("\nTraining sequential model with Curve merging...")
    model = model.to(device)
    old_model = None
    task_accuracies = []
    
    config = config or {}

    while True:
        train_loader, test_loader = seq_data.get_task_data()
        current_task = seq_data.current_task

        if current_task == 0:
            # First task: normal training
            train_model(
                config=config, 
                model=model, 
                train_loader=train_loader, 
                test_loader=test_loader, 
                epochs=1,
                device=device
            )
            old_model = copy_model_mlpnet(model)
        else:
            # Subsequent tasks: train and merge
            train_model(
                config=config, 
                model=model, 
                train_loader=train_loader, 
                test_loader=test_loader, 
                epochs=1,
                device=device
            )
            
            # Merge models using curve ensembling
            models = [old_model.to(device), model.to(device)]
            merged_model = curve_ensembling(
                config,
                models,
                train_loader,
                test_loader,
                device,
                num_classes,
                input_dim
            )
            
            # Update models
            old_model = copy_model_mlpnet(merged_model)
            model.load_state_dict(merged_model.state_dict())

        # Evaluate performance
        task_acc = evaluate_task_accuracies(
            model,
            seq_data.test_dataset,
            n_tasks=seq_data.n_tasks,
            device=device
        )
        task_accuracies.append(task_acc)

        if not seq_data.next_task():
            break

    return model, task_accuracies

def save_model(model, config, epoch, val_acc, test_acc, save_path):
    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'model_state_dict': model.state_dict(),
        'config': model.get_model_config()
    }, save_path)

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
