import math
import torch.nn as nn
import torch.nn.functional as F

import curves

__all__ = ['FCModel']

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import curves  # Ensure this is properly implemented and imported

from utils import hook

class FCModelBase(nn.Module):
    """
    FC deep model.
    The model has n hidden layers each consisting of linear network followed by
    ReLU activations as non-linearlity.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, bias=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param bias: If the linear elements should have bias
        """
        super(FCModelBase, self).__init__()
        self.num_classes = output_dim
        self.bias = bias
        self.channels = [input_dim] + hidden_dims + [output_dim]
        self.layers = []
        #self.layers = nn.ModuleList()
        for idx in range(1, len(self.channels)):
            cur_layer = [nn.Linear(self.channels[idx - 1], self.channels[idx], bias=self.bias)]
            #cur_layer = nn.ModuleList()
            #cur_layer.append(nn.Linear(self.channels[idx - 1], self.channels[idx], bias=self.bias))
            if idx + 1 < len(self.channels):
                cur_layer.append(nn.ReLU())
            seq_cur_layer = nn.Sequential(*cur_layer)
            self.layers.append(seq_cur_layer)
        self.layers_aggregated = nn.Sequential(*self.layers)

    def get_model_config(self):
        return {'input_dim': self.channels[0],
                'hidden_dims': self.channels[1:-1],
                'output_dim': self.channels[-1]}

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers_aggregated(x)

    @property
    def num_layers(self):
        return len(self.layers)

    @property
    def input_dim(self):
        return self.channels[0]

    def get_layer_weights(self, layer_num=1):
        assert 0 < layer_num <= self.num_layers
        # Returns the weights from the linear layer of this model
        return self.layers[layer_num - 1]._modules['0'].weight

    def get_activations(self, x, layer_num=0, pre_activations=False):
        # layer_num ranges from 1 to total layers.
        if layer_num == 0:
            return x.permute(1, 0)
        assert layer_num <= len(self.layers)
        if pre_activations:
            cur_hook = hook.Hook(self.layers[layer_num - 1]._modules['0'])
        else:
            cur_hook = hook.Hook(self.layers[layer_num - 1])
        self.eval()
        _ = self.forward(x)
        cur_hook.close()
        return cur_hook.output.transpose(0, 1).detach()

    def update_layer(self, layer, hidden_nodes, weights):
        # Updates an internal layer with custom hidden nodes and weights
        # Assumes that weights have appropriate dimension hidden_nodes x prev
        assert weights.size(0) == hidden_nodes
        assert weights.size(1) == self.channels[layer - 1]

        # Update current layer
        cur_layers = [nn.Linear(self.channels[layer - 1], hidden_nodes,
                                bias=self.bias)]
        if layer != self.num_layers:
            cur_layers.append(nn.ReLU())
        cur_layers[0].weight.data = weights.data
        self.layers[layer - 1] = nn.Sequential(*cur_layers)
        self.channels[layer] = hidden_nodes

        # Update Next layer
        if layer < self.num_layers:
            next_layer = [nn.Linear(self.channels[layer], self.channels[layer + 1],
                                    bias=self.bias)]
            if layer + 1 != self.num_layers:
                next_layer.append(nn.ReLU())
            self.layers[layer] = nn.Sequential(*next_layer)
        self.layers_aggregated = nn.Sequential(*self.layers)

class FCModelCurve(nn.Module):
    """
    Curve-based Fully Connected deep model.
    This model introduces curve layers that allow interpolation or merging.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, fix_points, bias=False):
        """
        :param input_dim: The input dimension
        :param hidden_dims: List of hidden dimensions for this model
        :param output_dim: Output dimension
        :param fix_points: Whether to fix points for the curve-based layers
        :param bias: If the linear elements should have bias
        """
        super(FCModelCurve, self).__init__()
        self.bias = bias
        self.channels = [input_dim] + hidden_dims + [output_dim]

        # Create curve-based layers
        self.layers = []
        for idx in range(1, len(self.channels)):
            cur_layer = curves.Linear(
                self.channels[idx - 1],
                self.channels[idx],
                fix_points=fix_points,
                bias=self.bias
            )
            self.layers.append(cur_layer)
        
        self.layers_aggregated = nn.ModuleList(self.layers)

    def forward(self, x, coeffs_t):
        """Forward pass through the curve-based network."""
        x = x.view(x.size(0), -1)

        for idx, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, coeffs_t))
        x = self.layers[-1](x, coeffs_t)  # Final layer without ReLU

        return F.log_softmax(x, dim=1)


class FCModel:
    base = FCModelBase
    curve = FCModelCurve    
    kwargs = {}