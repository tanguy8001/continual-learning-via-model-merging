import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict
import wandb

class CurveMLP(Module):
    def __init__(self, num_params, bias = True, hidden_dim= 32):
        super().__init__()  # Call parent's __init__ first
        self.hidden_dim = hidden_dim
        self.num_params = num_params

        # The MLP takes only t as input now
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=bias),  # Input is scalar t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params, bias=bias) # Output is correction vector
        )

    def forward(self,
                t: float,
                w0: torch.Tensor,   # shape (num_params,)
                w1: torch.Tensor    # shape (num_params,)
               ) -> torch.Tensor:
        """
        Returns:
          w_interp: torch.Tensor of shape (num_params,)
                     = (1−t)*w0 + t*w1 + t*(1−t)*MLP(t)
        """
        # Ensure t is on the same device and has the right shape for MLP input
        t_tensor = torch.as_tensor(t, dtype=w0.dtype, device=w0.device).view(1, 1)

        w0_flat = w0.view(-1)
        w1_flat = w1.view(-1)
        lin = (1.0 - t) * w0_flat + t * w1_flat        # shape (num_params,)

        # MLP takes only t as input
        mlp_out = self.mlp(t_tensor).view(-1) # Ensure output is flat (num_params,)

        corr = mlp_out * (t * (1.0 - t))  # shape (num_params,)
        return lin + corr


    def get_model_weights(self, model: Module):
        sd = model.state_dict()
        all_weights = torch.cat([
            v.view(-1)
            for k, v in sd.items()
            if "weight" in k
        ])
        return all_weights



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
        interpolation_points = torch.tensor([0.5], device=device)
        #interpolation_points = torch.rand(10, device=device)  # Generate 10 random values between 0 and 1
        flat1 = parameters_to_vector(model1.parameters()).to(device)
        flat2 = parameters_to_vector(model2.parameters()).to(device)
        specs = [(n, p.numel()) for n, p in model1.named_parameters()]

        run = wandb.init(
            entity = "Continual_Learning-DAL",
            project="Model Path Fusion for Continual Learning",

            config={
                "device": device, 
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "optimizer": "SGD",
                "buffer_size_in_batches": len(train_loader),
                #"interpolation_points": interpolation_points.tolist()
                "interpolation_points": interpolation_points,
                "dataset": config.dataset
            }
        )

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
                #interpolation_points =  torch.tensor([0.5])
                # interpolation_points =  torch.rand(6, device=device)
                # For each interpolation point
                for t in interpolation_points:
                    # Get the interpolated weights and perform forward pass in one go
                    w_interp = self(t, flat1, flat2)
                    interp_state = build_state_dict(w_interp, specs, model1)
                    outputs    = functional_call(model1, interp_state, (inputs,))
                    loss       = criterion(outputs, targets)
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
            
            wandb.log({
                "epoch": epoch, 
                "average Loss": avg_loss,
                "accuracy": accuracy
            })

def build_state_dict(w_flat: torch.Tensor,
                     specs: list[tuple[str,int]],
                     prototype_model: torch.nn.Module
                    ) -> OrderedDict:
    """
    Convert a flat parameter vector back into a OrderedDict matching prototype_model.
    
    Args:
      w_flat:    1D Tensor of length = sum(p.numel() for p in prototype_model.parameters())
      specs:     list of (name, numel) for each parameter in prototype_model.named_parameters()
      prototype_model: any nn.Module whose .state_dict() naming/param shapes you want to match
    
    Returns:
      OrderedDict{name → Tensor.view_as(original_param)}
    """
    state = OrderedDict()
    offset = 0
    named_params = dict(prototype_model.named_parameters())
    for name, numel in specs:
        chunk = w_flat[offset: offset + numel]
        offset += numel
        orig_param = named_params[name]
        state[name] = chunk.view_as(orig_param)
    return state