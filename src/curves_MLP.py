import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Subset 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import datasets, transforms
from torch.nn.utils.stateless import functional_call
from collections import OrderedDict, defaultdict
import wandb
from dataclasses import dataclass, field
from typing import List, Optional 
from enum import Enum
import random


class InterpolationType(Enum):
    STATIC  = "static"
    DYNAMIC = "dynamic"

@dataclass
class CurveConfigMLP:
    interpolation_type: InterpolationType
    interpolation_points: List[float]
    hidden_dim: int
    learning_rate: float
    momentum: float
    epochs: int 
    hidden_dims: List[int] = field(default_factory=lambda: [400,200,100])
    weight_decay: float = 5e-4

@dataclass
class ModelToMergeConfig:
    batch_size: int
    num_workers: int
    model_epochs: int
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [400,200,100])
    weight_decay: float = 5e-4
    epochs: int = 10
    learning_rate: float = 0.07


@dataclass
class TrainingBufferConfig:
    percentage: float

@dataclass
class ExperimentConfig:
    curve: CurveConfigMLP  
    model: ModelToMergeConfig 
    buffer: TrainingBufferConfig

 
def stratified_split(dataset, percentage: float, seed: int = 42):
    """
    Splits `dataset` into (subset_a, subset_b) so that each class in `dataset`
    contributes `percentage` of its samples to subset_a and the rest to subset_b.
    
    Args:
        dataset: a torch.utils.data.Dataset with a .targets or .labels attribute.
        pct: fraction in [0,1] of each class to go into subset_a.
        seed: random seed for reproducibility.
    
    Returns:
        (subset_a, subset_b) as torch.utils.data.Subset instances.
    """
    # 1) grab labels
    try:
        labels = dataset.targets
    except AttributeError:
        labels = dataset.labels  # some datasets use `.labels`
    
    # 2) bucket indices by class
    idx_by_class = defaultdict(list)
    for idx, lbl in enumerate(labels):
        idx_by_class[int(lbl)].append(idx)
    
    random.seed(seed)
    train_idxs, val_idxs = [], []
    for cls, idxs in idx_by_class.items():
        random.shuffle(idxs)
        split = int(len(idxs) * percentage)
        train_idxs.extend(idxs[:split])
        val_idxs.extend(idxs[split:])
    
    subset_a = Subset(dataset, train_idxs)
    subset_b = Subset(dataset, val_idxs)
    return subset_a, subset_b

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

    def forward(self,
                t: float,
                w0: torch.Tensor,   # shape (num_params,)
                w1: torch.Tensor    # shape (num_params,)
               ) -> torch.Tensor:
        """
        Returns:
          w_interp: torch.Tensor of shape (num_params,)
                     = (1−t)*w0 + t*w1 + t*(1−t)*MLP([ (1−t)*w0 + t*w1, t ])
        """
        t = w0.new_tensor(t)  # same dtype & device
        w0_flat = w0.view(-1)
        w1_flat = w1.view(-1)
        lin = (1.0 - t) * w0_flat + t * w1_flat        # shape (num_params,)

        t_vec  = t.unsqueeze(0)                 # (1,)
        mlp_in = torch.cat([w0_flat, w1_flat, t_vec], dim=0)  # (2*num_params + 1,)

        corr = self.mlp(mlp_in) * (t * (1.0 - t))  # shape (num_params,)
        return lin + corr


    def get_model_weights(self, model: Module):
        sd = model.state_dict()
        all_weights = torch.cat([
            v.view(-1)
            for k, v in sd.items()
            if "weight" in k
        ])
        return all_weights



    def fit(self, training_buffer: DataLoader , test_loader: DataLoader, cfg: CurveConfigMLP,
                model1: Module, model2: Module):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            opt = optim.SGD(self.mlp.parameters(),
                            lr=cfg.learning_rate,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)
            crit = nn.CrossEntropyLoss()

            # prepare weights once
            flat1 = parameters_to_vector(model1.parameters())
            flat2 = parameters_to_vector(model2.parameters())
            specs = [(n, p.numel()) for n, p in model1.named_parameters()]


            for epoch in range(cfg.epochs):
                tot_loss = tot_corr = tot_samples = 0
                for step, (x, y) in enumerate(training_buffer):
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = 0.0
                    for t in cfg.interpolation_points:
                        w_interp = self(t, flat1, flat2)
                        state = build_state_dict(w_interp, specs, model1)
                        out   = functional_call(model1, state, (x,))
                        loss += crit(out, y)
                        _, pred = out.max(1)
                        tot_corr += pred.eq(y).sum().item()
                        tot_samples += y.size(0)
                    loss = loss / len(cfg.interpolation_points)
                    loss.backward(); opt.step()
                    tot_loss += loss.item()
                    if step % 50 == 0:
                        print(f"E{epoch} S{step} L{loss:.4f}")
                acc = 100*tot_corr/tot_samples
                print(f"[Epoch {epoch}] loss={tot_loss/len(training_buffer):.4f} acc={acc:.2f}%")
                wandb.log({"epoch": epoch, "loss": tot_loss/len(training_buffer), "acc": acc})

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