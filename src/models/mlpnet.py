import math
import torch.nn as nn
import torch.nn.functional as F

import curves

__all__ = ['MlpNet']

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import curves  # Ensure this is properly implemented and imported

from utils import hook

class MlpNetBase(nn.Module):
    """Base Multi-Layer Perceptron Network implementation."""

    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 10, 
        width_ratio: float = 1.0,
        disable_bias: bool = True,
        enable_dropout: bool = False,
        hidden_dims: list = [400, 200, 100]  # New argument to match FCModel's structure
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.width_ratio = width_ratio if width_ratio > 0 else 1.0
        self.enable_dropout = enable_dropout
        self.hidden_dims = hidden_dims
        
        # Adjust hidden sizes based on width ratio
        scaled_sizes = [int(size / self.width_ratio) for size in self.hidden_dims]
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, scaled_sizes[0], bias=not disable_bias)
        self.fc2 = nn.Linear(scaled_sizes[0], scaled_sizes[1], bias=not disable_bias)
        self.fc3 = nn.Linear(scaled_sizes[1], scaled_sizes[2], bias=not disable_bias)
        self.fc4 = nn.Linear(scaled_sizes[2], num_classes, bias=not disable_bias)

        # Aggregate layers like in FCModel
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        self.layers_aggregated = nn.ModuleList(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        x = x.view(x.shape[0], -1)

        for idx, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if self.enable_dropout:
                x = F.dropout(x)

        x = self.fc4(x)  # Final layer without ReLU, output layer
        return F.log_softmax(x, dim=1)

    def get_model_config(self):
        return {'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.num_classes}

    @property
    def num_layers(self):
        return len(self.layers)

    #@property
    def input_dim(self):
        return self.input_dim
    
    def num_classes(self):
        return self.num_classes

    def get_layer_weights(self, layer_num: int) -> Tensor:
        """Returns the weights of the specified layer."""
        assert 0 < layer_num <= self.num_layers, "Invalid layer number"
        return self.layers[layer_num - 1].weight

    def get_activations(self, x: Tensor, layer_num: int = 0, pre_activations: bool = False) -> Tensor:
        """Returns the activations or pre-activations of a specified layer."""
        assert 0 <= layer_num <= self.num_layers, "Invalid layer number"
        if layer_num == 0:
            return x.permute(1, 0)  # Input layer

        # Hook into the specified layer for activations
        layer = self.layers[layer_num - 1]
        if pre_activations:
            cur_hook = hook.Hook(layer[0])  # Hook on the linear part of the layer
        else:
            cur_hook = hook.Hook(layer)  # Hook on the entire layer
        self.eval()
        _ = self.forward(x)  # Forward pass to collect activations
        cur_hook.close()
        return cur_hook.output.transpose(0, 1).detach()

    def update_layer(self, layer: int, hidden_nodes: int, weights: Tensor) -> None:
        """Updates an internal layer with custom hidden nodes and weights."""
        assert weights.size(0) == hidden_nodes, "Weight dimensions don't match"
        assert weights.size(1) == self.input_dim if layer == 1 else self.layers[layer - 2].out_features, "Invalid weight dimensions"

        # Update the current layer with the new number of hidden nodes
        current_layer = nn.Linear(self.layers[layer - 1].in_features, hidden_nodes)
        current_layer.weight.data = weights.data
        self.layers[layer - 1] = current_layer
        
        # Update next layer if applicable
        if layer < self.num_layers:
            next_layer = nn.Linear(hidden_nodes, self.layers[layer].out_features)
            self.layers[layer] = next_layer

        self.layers_aggregated = nn.ModuleList(self.layers)


class MlpNetCurve(nn.Module):
    """Curve-based version of the MLP network for model merging."""
    
    def __init__(
        self, 
        num_classes: int,
        fix_points: bool,
        width_ratio: float = 1.0,
        input_dim: int = 784,
        disable_bias: bool = True,
        enable_dropout: bool = False
    ):
        super().__init__()
        
        self.width_ratio = width_ratio if width_ratio > 0 else 1.0
        self.enable_dropout = enable_dropout
        
        # Define layer sizes
        hidden_sizes = [400, 200, 100]
        scaled_sizes = [int(size / self.width_ratio) for size in hidden_sizes]
        
        # Create curve-based layers
        self.fc1 = curves.Linear(input_dim, scaled_sizes[0], fix_points=fix_points, bias=not disable_bias)
        self.fc2 = curves.Linear(scaled_sizes[0], scaled_sizes[1], fix_points=fix_points, bias=not disable_bias)
        self.fc3 = curves.Linear(scaled_sizes[1], scaled_sizes[2], fix_points=fix_points, bias=not disable_bias)
        self.fc4 = curves.Linear(scaled_sizes[2], num_classes, fix_points=fix_points, bias=not disable_bias)

    def forward(self, x: torch.Tensor, coeffs_t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the curve-based network."""
        x = x.view(x.shape[0], -1)
        
        for layer in [self.fc1, self.fc2, self.fc3]:
            x = F.relu(layer(x, coeffs_t))
            if self.enable_dropout:
                x = F.dropout(x)
                
        x = self.fc4(x, coeffs_t)
        return F.log_softmax(x, dim=1)

    
class MlpNet(nn.Module):
    base = MlpNetBase
    curve = MlpNetCurve
    kwargs = {}