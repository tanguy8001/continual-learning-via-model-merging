from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
    gradient_clipping: bool = False
    hidden_dims: List[int] = field(default_factory=lambda: [400,200,100])
    weight_decay: float = 5e-4
    use_positional_encoding: bool = False
    pos_encoding_dim: int = 32
    pos_encoding_freq: float = 1.0
    t_only_mode: bool = False

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
    momentum: float = 0.9


@dataclass
class TrainingBufferConfig:
    percentage: float

@dataclass
class ExperimentConfig:
    curve: CurveConfigMLP  
    model: ModelToMergeConfig 
    buffer: TrainingBufferConfig

def split_classes_with_sampled_fraction(
    dataset,
    class_fraction: float = 0.5,
    sample_fraction: float = 0.9,
    shuffle: bool = True,
    seed: Optional[int] = None
):
    """
    1) Partition classes into two groups: A gets the first `class_fraction` of classes,
       B gets the rest (so here class_fraction=0.5 → half/half).
    2) From each class in each group, take `sample_fraction` of its indices (e.g. 0.8 → 80%).

    Returns:
        subset_A, subset_B : Subsets containing 80% of group-A classes and 80% of group-B classes.
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Dataset must have .targets or .labels")
    
    labels = np.array(labels)

    classes = np.unique(labels)
    rng = np.random.RandomState(seed)
    if shuffle:
        rng.shuffle(classes)

    n_A = int(len(classes) * class_fraction)
    classes_A = classes[:n_A]
    classes_B = classes[n_A:]

    # 4. sample indices per class
    idx_A, idx_B = [], []
    for cls in classes_A:
        cls_idx = np.where(labels == cls)[0]
        if shuffle:
            rng.shuffle(cls_idx)
        cut = int(len(cls_idx) * sample_fraction)
        idx_A.extend(cls_idx[:cut])
        idx_B.extend(cls_idx[cut:])
    for cls in classes_B:
        cls_idx = np.where(labels == cls)[0]
        if shuffle:
            rng.shuffle(cls_idx)
        cut = int(len(cls_idx) * sample_fraction)
        idx_B.extend(cls_idx[:cut])
        idx_A.extend(cls_idx[cut:])

    # 5. final shuffle
    if shuffle:
        rng.shuffle(idx_A)
        rng.shuffle(idx_B)

    # 6. build Subsets
    subset_A = Subset(dataset, idx_A)
    subset_B = Subset(dataset, idx_B)
    return subset_A, subset_B


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
    # Fix the boolean evaluation issue with tensors
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        raise ValueError("Dataset must have .targets or .labels")
    
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
    def __init__(self, in_features, out_features, bias = True, hidden_dim= 32, t_only_mode=False, num_terms=3, 
                 use_positional_encoding=False, pos_encoding_dim=32, pos_encoding_freq=1.0):
        super().__init__()  # Call parent's __init__ first
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.out_features = out_features 
        self.t_only_mode = t_only_mode
        self.num_terms = num_terms
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_dim = pos_encoding_dim
        self.pos_encoding_freq = pos_encoding_freq
        
        # Create multiple MLPs for different polynomial terms
        self.mlps = nn.ModuleList()
        
        for i in range(num_terms):
            if t_only_mode:
                # Determine input dimension based on positional encoding
                if use_positional_encoding:
                    input_dim = pos_encoding_dim
                else:
                    input_dim = 1  # Only t
                
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim, bias=bias),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=bias),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_features, bias=bias)
                )
            else:
                # Original mode: weights + t as input
                if use_positional_encoding:
                    input_dim = in_features + pos_encoding_dim  # weights + positional encoding
                else:
                    input_dim = in_features + 1  # weights + t
                
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim, bias=bias),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=bias),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_features, bias=bias)
                )
            self.mlps.append(mlp)

    def positional_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to scalar t.
        
        Args:
            t: Scalar tensor of shape (1,) or (batch_size,)
            
        Returns:
            Positional encoding of shape (pos_encoding_dim,) or (batch_size, pos_encoding_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)  # Add batch dimension if scalar
            
        # Create frequency bands
        freqs = torch.arange(0, self.pos_encoding_dim, 2, device=t.device, dtype=t.dtype)
        freqs = self.pos_encoding_freq * (2 ** (freqs / self.pos_encoding_dim))
        
        # Apply sin/cos encoding
        sin_enc = torch.sin(freqs * t.unsqueeze(-1))  # (batch_size, pos_encoding_dim//2)
        cos_enc = torch.cos(freqs * t.unsqueeze(-1))  # (batch_size, pos_encoding_dim//2)
        
        # Interleave sin and cos
        encoding = torch.zeros(t.size(0), self.pos_encoding_dim, device=t.device, dtype=t.dtype)
        encoding[:, 0::2] = sin_enc
        encoding[:, 1::2] = cos_enc
        
        return encoding

    def forward(self,
                t: torch.Tensor,  # Changed from float to torch.Tensor
                w0: torch.Tensor,   # shape (num_params,)
                w1: torch.Tensor    # shape (num_params,)
               ) -> torch.Tensor:
        """
        Returns:
          w_interp: torch.Tensor of shape (num_params,)
                     = (1−t)*w0 + t*w1 + sum of multiple correction terms with different polynomial bases
        """
        if isinstance(t, (int, float)):
            t = w0.new_tensor(t)  # same dtype & device
        w0_flat = w0.view(-1)
        w1_flat = w1.view(-1)
        lin = (1.0 - t) * w0_flat + t * w1_flat        # shape (num_params,)

        # Define different polynomial bases for more freedom
        # These are orthogonal-like polynomials that give different behaviors
        poly_terms = [
            t * (1.0 - t),                    # t - t² (quadratic)
            t * (1.0 - t) * (1.0 - 2.0 * t),  # t - 3t² + 2t³ (cubic) 
            t * (1.0 - t) * (1.0 - 2.0 * t) * (1.0 - 3.0 * t),  # t - 6t² + 11t³ - 6t⁴ (quartic)
        ]
        
        # Alternative: Fourier basis for even more freedom
        # This can create much more complex curves
        fourier_terms = [
            torch.sin(2 * torch.pi * t),
            torch.cos(2 * torch.pi * t), 
            torch.sin(4 * torch.pi * t),
            torch.cos(4 * torch.pi * t),
            torch.sin(6 * torch.pi * t),
            torch.cos(6 * torch.pi * t),
        ]
        
        corr = torch.zeros_like(lin)
        
        for i, mlp in enumerate(self.mlps):
            if i < len(poly_terms):
                if self.t_only_mode:
                    # Use positional encoding or just t as input to MLP
                    if self.use_positional_encoding:
                        t_encoded = self.positional_encoding(t)  # (pos_encoding_dim,)
                        corr += mlp(t_encoded) * poly_terms[i]
                    else:
                        # Only use t as input to MLP
                        t_vec = t.unsqueeze(0)  # (1,)
                        corr += mlp(t_vec) * poly_terms[i]
                else:
                    # Original mode: use weights + t as input
                    if self.use_positional_encoding:
                        t_encoded = self.positional_encoding(t)  # (pos_encoding_dim,)
                        mlp_in = torch.cat([w0_flat, w1_flat, t_encoded], dim=0)  # (2*num_params + pos_encoding_dim,)
                    else:
                        t_vec = t.unsqueeze(0)                 # (1,)
                        mlp_in = torch.cat([w0_flat, w1_flat, t_vec], dim=0)  # (2*num_params + 1,)
                    corr += mlp(mlp_in) * poly_terms[i]
        
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
            opt = optim.SGD(self.mlps.parameters(),
                            lr=cfg.learning_rate,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)
            crit = nn.CrossEntropyLoss()

            # prepare weights once
            flat1 = parameters_to_vector(model1.parameters()).detach()
            flat2 = parameters_to_vector(model2.parameters()).detach()
            print("  w0  min/max:", flat1.min().item(), flat1.max().item())
            print("  w1  min/max:", flat2.min().item(), flat2.max().item())

            specs = [(n, p.numel()) for n, p in model1.named_parameters()]
            

            for epoch in range(cfg.epochs):
                tot_loss = tot_corr = tot_samples = 0
                for step, (x, y) in enumerate(training_buffer):
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = torch.tensor(0.0, device=device, requires_grad=True)  # Ensure loss is a tensor
                    
                    interpolationPoints = cfg.interpolation_points 
                    if cfg.interpolation_type is InterpolationType.DYNAMIC: 
                        interpolationPoints = [random.random() for _ in range(5)]

                    for t in interpolationPoints: 
                        w_interp = self(t, flat1, flat2)
                        state = build_state_dict(w_interp, specs, model1)
                        out   = functional_call(model1, state, (x,))
                        loss = loss + crit(out, y)
                        _, pred = out.max(1)
                        tot_corr += pred.eq(y).sum().item()
                        tot_samples += y.size(0)
        
                    loss = loss / len(cfg.interpolation_points)
                    loss.backward(); 
                    if cfg.gradient_clipping == True:
                        torch.nn.utils.clip_grad_norm_(self.mlps.parameters(), max_norm=1.0)
                    opt.step()
                    tot_loss += loss.item()
                    if step % 50 == 0:
                        print(f"E{epoch} S{step} L{loss:.4f}")
                acc = 100*tot_corr/tot_samples
                print(f"[Epoch {epoch}] loss={tot_loss/len(training_buffer):.4f} acc={acc:.2f}%")
                # wandb.log({"epoch": epoch, "loss": tot_loss/len(training_buffer), "acc": acc})
            
            # Evaluate the final merged model on the total buffer
            print("\nEvaluating final merged model on total buffer...")
            self.eval()
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    
                    # Evaluate at t=0.5 (middle of interpolation)
                    t = 0.5
                    w_interp = self(t, flat1, flat2)
                    state = build_state_dict(w_interp, specs, model1)
                    out = functional_call(model1, state, (x,))
                    
                    loss = crit(out, y)
                    total_loss += loss.item() * x.size(0)
                    
                    _, pred = out.max(1)
                    total_correct += pred.eq(y).sum().item()
                    total_samples += y.size(0)
            
            final_acc = 100 * total_correct / total_samples
            final_loss = total_loss / total_samples
            print(f"Final merged model performance on total buffer:")
            print(f"  Loss: {final_loss:.4f}")
            print(f"  Accuracy: {final_acc:.2f}%")
            # wandb.log({"final_loss": final_loss, "final_acc": final_acc})
            
            return final_loss, final_acc

    def evaluate_curve(self, test_loader: DataLoader, model1: Module, model2: Module, 
                      interpolation_points: Optional[List[float]] = None) -> dict:
        """
        Evaluate the merged model at different interpolation points.
        
        Args:
            test_loader: DataLoader for evaluation
            model1: First model
            model2: Second model  
            interpolation_points: List of t values to evaluate at (default: [0.0, 0.25, 0.5, 0.75, 1.0])
            
        Returns:
            Dictionary with results for each interpolation point
        """
        if interpolation_points is None:
            interpolation_points = [0.0, 0.25, 0.5, 0.75, 1.0]
            
        device = next(self.parameters()).device
        self.eval()
        
        # prepare weights once
        flat1 = parameters_to_vector(model1.parameters()).detach()
        flat2 = parameters_to_vector(model2.parameters()).detach()
        specs = [(n, p.numel()) for n, p in model1.named_parameters()]
        crit = nn.CrossEntropyLoss()
        
        results = {}
        
        for t in interpolation_points:
            total_correct = 0
            total_samples = 0
            total_loss = 0.0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    
                    w_interp = self(t, flat1, flat2)
                    state = build_state_dict(w_interp, specs, model1)
                    out = functional_call(model1, state, (x,))
                    
                    loss = crit(out, y)
                    total_loss += loss.item() * x.size(0)
                    
                    _, pred = out.max(1)
                    total_correct += pred.eq(y).sum().item()
                    total_samples += y.size(0)
            
            acc = 100 * total_correct / total_samples
            loss = total_loss / total_samples
            
            results[f"t={t:.2f}"] = {
                "accuracy": acc,
                "loss": loss
            }
            
            print(f"t={t:.2f}: Loss={loss:.4f}, Accuracy={acc:.2f}%")
        
        return results

    def evaluate_curve_dimensionality(self, model1: Module, model2: Module, num_points: int = 1000, plot: bool = True, variance_threshold: float = 0.99):
        """
        Evaluates the intrinsic dimensionality of the curve between two models using PCA.
        Args:
            model1: First model (torch.nn.Module)
            model2: Second model (torch.nn.Module)
            num_points: Number of points to sample along the curve
            plot: Whether to plot the cumulative explained variance
            variance_threshold: Threshold for cumulative variance to determine dimensionality
        Returns:
            n_components: Number of components to reach variance_threshold
            explained_variance: Array of explained variance ratios
        """
        flat1 = parameters_to_vector(model1.parameters()).detach()
        flat2 = parameters_to_vector(model2.parameters()).detach()
        X = sample_curve_points(self, flat1, flat2, num_points=num_points)
        n_components, explained_variance = analyze_curve_dimensionality(X, plot=plot, variance_threshold=variance_threshold)
        return n_components, explained_variance

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

def sample_curve_points(curve_mlp, flat1, flat2, num_points=1000):
    """
    Samples points along the curve defined by curve_mlp between flat1 and flat2.
    Args:
        curve_mlp: The MLP curve object (callable: t, flat1, flat2 -> weights)
        flat1: Flattened weights of model 1 (torch.Tensor)
        flat2: Flattened weights of model 2 (torch.Tensor)
        num_points: Number of points to sample along the curve
    Returns:
        X: np.ndarray of shape (num_points, num_weights)
    """
    t_values = np.linspace(0, 1, num_points)
    curve_points = []
    for t in t_values:
        t_tensor = torch.tensor(t, dtype=flat1.dtype, device=flat1.device)
        w_curve = curve_mlp(t_tensor, flat1, flat2).detach().cpu().numpy()
        curve_points.append(w_curve)
    X = np.stack(curve_points)
    return X


def analyze_curve_dimensionality(X, plot=True, variance_threshold=0.99):
    """
    Runs PCA on the sampled curve points and analyzes intrinsic dimensionality.
    Args:
        X: np.ndarray of shape (num_points, num_weights)
        plot: Whether to plot cumulative explained variance
        variance_threshold: Threshold for cumulative variance to determine dimensionality
    Returns:
        n_components: Number of components to reach variance_threshold
        explained_variance: Array of explained variance ratios
    """
    import numpy as np

    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)

    print(f"Explained variance of first component: {explained_variance[0]:.6f}")
    print(f"Explained variance of second component: {explained_variance[1]:.6f}")
    print(f"Explained variance of first two components: {np.sum(explained_variance[:2]):.6f}")

    n_components = np.argmax(cumulative >= variance_threshold) + 1
    print(f"Number of components to explain {variance_threshold*100:.1f}% variance: {n_components}")

    if plot:
        plt.figure()
        plt.plot(np.arange(1, len(cumulative)+1), cumulative*100, marker='o')
        plt.xlabel('Number of principal components')
        plt.ylabel('Cumulative explained variance (%)')
        plt.title('Intrinsic dimensionality of the curve')
        plt.grid()
        plt.show()

    return n_components, explained_variance