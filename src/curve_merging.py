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

#TODO: renommer curve_fusion.py 

class CurveFusion:

    def __init__(self, args, base_models, target_model, data):
        self.args = args
        self.base_models = base_models
        self.target_model = target_model
        self.data = data

    def fuse(self):
        '''
        Directly modifies the target_model with its new weights.
        '''

        logging.info("Starting curve model fusion")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in self.base_models:
            model.to(device)
        #self.target_model.to(device)
        #if self.data is not None:
        #    self.data = self.data.cuda()

        self.target_model = curve_ensembling(
            config=CurveConfig(),
            models=self.base_models,
            target_model=self.target_model,
            train_loader=self.data['train'],
            test_loader=self.data['test'],
            device=device,
            num_classes=self.target_model.num_classes,
            input_dim=self.target_model.input_dim,
        ).to(device)

        logging.info('Curve model fusion completed.')

@dataclass
class CurveConfig:
    """Configuration for curve merging parameters."""
    base_dir: Path = Path.cwd()
    transform: str = "MLPNET"
    model: str = "FCModel"
    dataset: str = "MNIST"
    input_dim: int = 3072 if dataset == "CIFAR10" else 784
    #hidden_dims: List[int] = field(default_factory=lambda: [400, 200, 100])
    #hidden_dims: List[int] = field(default_factory=lambda: [800, 400, 200])
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    output_dim: int = 10
    epochs: int = 10 # epochs for the curve training, not models!
    model_epochs: int = 10
    model_learning_rate: float = 0.007
    learning_rate: float = 0.07
    weight_decay: float = 5e-4
    momentum: float = 0.9
    num_classes: int = 10
    num_bends: int = 3
    num_workers: int = 2
    curve: str = "Bezier"
    num_points: int = 61
    batch_size: int = 128
    grid_points: int = 21
    num_workers: int = 1
    fix_start: bool = True
    fix_end: bool = True
    cifar_class: str = "dog"
    test_digit: int = 4


def train_model(
    config: CurveConfig,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    device: str = "cpu",
    learning_rate: float = 0.07
) -> None:
    """
    Train a model using SGD optimizer and cross-entropy loss.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Learning rate for SGD optimizer
    """
    #print(f"Learning rate used: {learning_rate}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        momentum=config.momentum,
        lr=learning_rate, 
        weight_decay=config.weight_decay if config.curve is None else 0.0
    )
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            lr = learning_rate_schedule(config.learning_rate, epoch, config.epochs)
            adjust_learning_rate(optimizer, lr)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

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
    config: CurveConfig,
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
    architecture = getattr(mods, config.model)
    architecture.kwargs['input_dim'] = config.input_dim
    architecture.kwargs['hidden_dims'] = config.hidden_dims
    architecture.kwargs['output_dim'] = config.output_dim
    curve = getattr(curves, config.curve)
    
    curve_model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        config.num_bends,
        config.fix_start,
        config.fix_end,
        architecture_kwargs=architecture.kwargs,
    ).to(device)

    curve_model.import_base_parameters(models[0], 0)
    curve_model.import_base_parameters(models[1], config.num_bends - 1)
    
    # Train the curve model
    train_model(config, curve_model, train_loader, test_loader, learning_rate=config.learning_rate, epochs=config.epochs)
    model_path = "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints"
    final_save_path = os.path.join(model_path, 'final_curve_model.pth')
    save_model(curve_model, config, config.epochs, -1, -1, final_save_path)
    
    ## Create merged model
    #merged_model = architecture.base(
    #    input_dim=input_dim, 
    #    num_classes=num_classes, 
    #    **architecture.kwargs
    #).to(device)
    
    # Sample weights from middle of curve
    steps = np.linspace(0.0, 1.0, config.num_points)
    middle_step = steps[len(steps) // 2]
    fusion_weights = curve_model.weights(torch.tensor([middle_step]))
    
    # Update merged model parameters
    offset = 0
    for parameter in target_model.parameters():
        size = np.prod(parameter.size())
        value = fusion_weights[offset:offset + size].reshape(parameter.size())
        parameter.data.copy_(torch.from_numpy(value))
        offset += size

    #model_path = "/home/tdieudonne/dl3/src/tlp_model_fusion/checkpoints"
    #final_save_path = os.path.join(model_path, 'final_curve_fusion_model.pth')
    #config = CurveConfig()
    #save_model(target_model, config, config.epochs, val_acc, test_acc, final_save_path)
    
    return target_model


def train_merging_curve(
    seq_data,
    model: nn.Module,
    device: str = "cpu",
    num_classes: int = 10,
    input_dim: int = 784,
    config: Optional[CurveConfig] = None
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
    
    config = config or CurveConfig()

    while True:
        train_loader, test_loader = seq_data.get_task_data()
        current_task = seq_data.current_task

        if current_task == 0:
            # First task: normal training
            train_model(config, model, train_loader, test_loader, epochs=1, device=device)
            old_model = copy_model_mlpnet(model)
        else:
            # Subsequent tasks: train and merge
            train_model(config, model, train_loader, test_loader, epochs=1, device=device)
            
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