import torch
import numpy as np
import random
import os
import copy

from models.mlpnet import MlpNetBase


def copy_model(
    model,
    input_dim,
    num_classes,
):
    """Create a deep copy of model weights"""

    copy = MlpNetBase(
        input_dim=input_dim,
        num_classes=num_classes,
    )
    copy.load_state_dict(
        {name: param.clone() for name, param in model.state_dict().items()}
    )
    return copy


def load_models(dir_path,num_models,model,input_dim,num_classes):
    models = []
    for i in range(num_models):
        curr_model = copy_model(model,input_dim,num_classes)
        path = os.path.join(dir_path,f"model_{i}.checkpoint")
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, "cpu")
            ),
        )
        model_state_dict = state["model_state_dict"]
        curr_model.load_state_dict(model_state_dict)
        models.append(curr_model)

    return models


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(43)
