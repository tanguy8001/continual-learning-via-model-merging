import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.optim as optim
import copy


class CurveMLP(Module):
    def __init__(self, in_features, out_features, bias = True, hidden_dim= 32):
        super().__init__()  # Call parent's __init__ first
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.out_features = out_features 
        
        # The MLP takes the interpolated weights and t as input
        self.mlp = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim, bias=bias),  # +1 for t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features, bias=bias)
        )

    def forward(self, t: float, W1: torch.Tensor, W2: torch.Tensor):
        """
        t: Python float scalar in [0,1]
        """
        # Convert t to tensor and ensure it's on the right device
        t_tensor = torch.tensor(t, dtype=W1.dtype, device=W1.device)
        
        # Unroll both weight matrices into vectors
        W1_unrolled = W1.view(-1)  # shape (total_params,)
        W2_unrolled = W2.view(-1)  
        
        # Linear interpolation between unrolled weights
        lin = (1.0 - t_tensor) * W1_unrolled + t_tensor * W2_unrolled
        
        # Create t_broadcast with correct shape
        t_broadcast = t_tensor.expand(1)  # shape (1,)
        
        # Concatenate unrolled weights with t
        mlp_in = torch.cat([W1_unrolled, W2_unrolled, t_broadcast], dim=0)  # shape (2*total_params + 1,)
        
        # Get correction from MLP and scale by t*(1-t)
        corr_unrolled = (t_tensor * (1.0 - t_tensor)) * self.mlp(mlp_in)
        
        # Reshape correction back to original matrix shape
        corr = corr_unrolled.view(W1.shape)
        
        # Reshape linear interpolation back to matrix shape and add correction
        return lin.view(W1.shape) + corr

    def interpolate_weights(self, model1, model2, t):
        new_model = copy.deepcopy(model1)
        # 1) flatten *all* parameters of both models
        flat1 = torch.cat([p.view(-1) for p in model1.parameters()])
        flat2 = torch.cat([p.view(-1) for p in model2.parameters()])
    
        flat_interp = self.forward(float(t), flat1, flat2)
    
        # 3) split and write back into new_model
        offset = 0
        for p in new_model.parameters():
            n = p.numel()
            chunk = flat_interp[offset: offset + n]
            p.data.copy_(chunk.view_as(p))
            offset += n
    
        return new_model

    def fit(self, train_loader, test_loader, config, model1: Module, model2: Module, device="cpu"):
        self.to(device)
        optimizer = optim.SGD(
            self.mlp.parameters(),           # ← only update mlp
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Define interpolation points
        interpolation_points = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0], device=device)

        for epoch in range(config.epochs):
            # Training phase
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Initialize batch loss
                batch_loss = 0.0
                
                # For each interpolation point
                for t in interpolation_points:
                    # Get the interpolated model
                    interpolated_model = self.interpolate_weights(model1, model2, t)
                    interpolated_model.to(device)
                    
                    # Forward pass through interpolated model
                    outputs = interpolated_model(inputs)
                    loss = criterion(outputs, targets)
                    batch_loss += loss
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    total_samples += targets.size(0)
                
                # Average loss over interpolation points
                batch_loss = batch_loss / len(interpolation_points)
                
                # Backward pass
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {batch_loss.item():.4f}')

            # Print epoch statistics
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * total_correct / total_samples
            print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # For each interpolation point
                    for t in interpolation_points:
                        interpolated_model = self.interpolate_weights(model1, model2, t)
                        interpolated_model.to(device)
                        
                        outputs = interpolated_model(inputs)
                        val_loss += criterion(outputs, targets).item()
                        
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(targets).sum().item()
                        val_total += targets.size(0)
                
                val_loss = val_loss / (len(test_loader) * len(interpolation_points))
                val_accuracy = 100. * val_correct / val_total
                print(f'Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
                print('-' * 50)
