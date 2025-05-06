"""
Trains two base models (A and B) on disjoint data splits (e.g., MNIST or CIFAR10).
Then computes and saves two types of connecting curves between them:
1. A Bezier curve trained using `curve_merging.curve_ensembling`.
2. An MLP-based curve trained using `curves_MLP.CurveMLP`.

Saves Model A, Model B, the Bezier CurveNet, and the CurveMLP checkpoints
to a specified directory for later use (e.g., by `plot_loss_surface.py`).
"""

import os
import torch
import argparse
import copy
from tqdm import tqdm
import numpy as np
import yaml # Import YAML library

from data import double_loaders, create_fused_loader
from models import fcmodel
from curve_merging import (
    train_model as train_base_model, # Rename to avoid clash
    curve_ensembling,
    evaluate_model
)
from curves_MLP import CurveMLP

# Helper function to save models consistently
def save_checkpoint(model, config_dict, save_path, is_curve_mlp=False, curve_mlp_hidden_dim=None):
    """Saves model state_dict along with configuration."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config_dict # Store relevant config
    }
    # Add specific CurveMLP config if needed
    if is_curve_mlp and curve_mlp_hidden_dim is not None:
         checkpoint['config']['hidden_dim'] = curve_mlp_hidden_dim

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def main(config): # Accept loaded config dictionary
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    data_path = os.path.join(config['data_path'], config['dataset'])
    os.makedirs(data_path, exist_ok=True)
    
    # --- Data Loading ---
    data_loaders, num_classes = double_loaders(
        dataset=config['dataset'],
        path=config['data_path'], # Use root data path
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        transform_name=config['transform'], # Get transform based on dataset/model
        digit=config['mnist_digit'], # Specific digit for MNIST split
        cifar_class=config['cifar_class'] # Specific class for CIFAR split
    )

    fused_loader = create_fused_loader(
        data_loaders['trainA'],
        data_loaders['trainB'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Determine input dimension based on dataset
    if config['dataset'] == 'MNIST':
        input_dim = 784
    elif config['dataset'] == 'CIFAR10':
        input_dim = 3 * 32 * 32 # 3072
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    # Allow override from config if provided
    input_dim = config['input_dim'] if config['input_dim'] else input_dim
    output_dim = num_classes

    # --- Model Initialization ---
    model_config = {
        'input_dim': input_dim,
        'hidden_dims': config['hidden_dims'],
        'output_dim': output_dim
    }
    model_A = fcmodel.FCModelBase(**model_config).to(device)
    model_B = fcmodel.FCModelBase(**model_config).to(device)

    model_A_path = os.path.join(config['checkpoint_dir'], f"{config['dataset'].lower()}_model_A.pth")
    model_B_path = os.path.join(config['checkpoint_dir'], f"{config['dataset'].lower()}_model_B.pth")

    # --- Train/Load Base Models ---
    # Define a base config for training (can be simple if train_base_model doesn't use much)
    # Use a simple Namespace or dict for base_train_config if CurveConfig isn't needed
    base_train_config = argparse.Namespace(
        learning_rate=config['base_model_lr'],
        momentum=config.get('momentum', 0.9), # Use .get for safety
        weight_decay=config.get('weight_decay', 5e-4)
        # Add other fields train_base_model might need from config
    )

    if os.path.exists(model_A_path) and not config['force_retrain']:
        print(f"Loading Model A from {model_A_path}")
        checkpoint_A = torch.load(model_A_path, map_location=device)
        # TODO: Verify if saved config in checkpoint matches current model_config if needed
        model_A.load_state_dict(checkpoint_A['model_state_dict'])
    else:
        print("Training Model A...")
        train_base_model(
            config=base_train_config, # Pass necessary config fields
        model=model_A,
        train_loader=data_loaders['trainA'],
        test_loader=data_loaders['test'],
            epochs=config['base_model_epochs'], # Use config value
        device=device,
            learning_rate=config['base_model_lr'] # Use config value
        )
        save_checkpoint(model_A, model_config, model_A_path)

    if os.path.exists(model_B_path) and not config['force_retrain']:
        print(f"Loading Model B from {model_B_path}")
        checkpoint_B = torch.load(model_B_path, map_location=device)
        model_B.load_state_dict(checkpoint_B['model_state_dict'])
    else:
        print("Training Model B...")
        train_base_model(
            config=base_train_config,
        model=model_B,
        train_loader=data_loaders['trainB'],
        test_loader=data_loaders['test'],
            epochs=config['base_model_epochs'], # Use config value
            device=device,
            learning_rate=config['base_model_lr'] # Use config value
        )
        save_checkpoint(model_B, model_config, model_B_path)

    # Evaluate base models
    acc_A = evaluate_model(model_A, data_loaders['test'], device)
    acc_B = evaluate_model(model_B, data_loaders['test'], device)
    print(f"\nModel A Final Accuracy: {acc_A:.2f}%")
    print(f"Model B Final Accuracy: {acc_B:.2f}%")

    # --- Bezier Curve Training ---
    print("\nTraining Bezier Curve (CurveNet)...")
    bezier_curve_path = os.path.join(config['checkpoint_dir'], f"{config['dataset'].lower()}_bezier_curve.pth")

    # Create a target model instance for curve_ensembling
    target_model_bezier = fcmodel.FCModelBase(**model_config).to(device)

    # Prepare config dictionary for curve_ensembling
    bezier_train_config_dict = {
        'model': config['model'],
        'epochs': config['bezier_epochs'],
        'learning_rate': config['bezier_lr'],
        'momentum': config['optimizer_momentum'],
        'weight_decay': config['optimizer_weight_decay'],
        'num_bends': config['bezier_num_bends'],
        'curve': config['curve'],
        'fix_start': config['bezier_fix_start'],
        'fix_end': config['bezier_fix_end'],
        'num_points': config['bezier_num_points'],
        'curve_checkpoint_path': bezier_curve_path,
        'dataset': config['dataset'],
        'input_dim': input_dim,
        'hidden_dims': config['hidden_dims'],
        'output_dim': output_dim,
        'batch_size': config['batch_size']
    }

    bezier_curve_model = curve_ensembling(
        config=config,
        models=[model_A, model_B],
        target_model=target_model_bezier,
        train_loader=fused_loader,
        test_loader=data_loaders['test'],
        device=device,
        num_classes=output_dim,
        input_dim=input_dim
    )

    acc_bezier_final = evaluate_model(bezier_curve_model, data_loaders['test'], device)
    print(f"Bezier Curve (Final State) Accuracy: {acc_bezier_final:.2f}%")

    # --- MLP Curve Training ---
    print("\nTraining MLP Curve (CurveMLP)...")
    mlp_curve_path = os.path.join(config['checkpoint_dir'], f"{config['dataset'].lower()}_mlp_curve.pth")

    # Get flat parameter vectors
    with torch.no_grad():
        w0 = torch.cat([p.view(-1) for p in model_A.parameters()]).detach()
        w1 = torch.cat([p.view(-1) for p in model_B.parameters()]).detach()
    num_params = w0.numel()

    # Initialize CurveMLP
    curve_mlp = CurveMLP(
        num_params=num_params,
        bias=True,
        hidden_dim=config['mlp_hidden_dim']
    ).to(device)
    
    # Configure CurveMLP training
    mlp_train_config = {
        'epochs': config['mlp_epochs'],
        'learning_rate': config['mlp_lr'],
        'momentum': config['optimizer_momentum'], # Use config value
        'weight_decay': config['optimizer_weight_decay'], # Use config value
        'batch_size': config['batch_size'],
        'save_path': mlp_curve_path,
        'dataset': config['dataset'],
        'input_dim': input_dim,
        'hidden_dims': config['hidden_dims'], # Pass base model dims
        'output_dim': output_dim
    }
    
    history_mlp = curve_mlp.fit(
        train_loader=fused_loader,
        test_loader=data_loaders['test'],
        config=mlp_train_config, # Pass the dictionary
        model1=model_A,
        model2=model_B
    )

    # Evaluate the final MLP curve model (typically at t=0.5)
    curve_mlp.eval()
    with torch.no_grad():
        t_mid = torch.tensor([0.5], device=device)
        w_mid_mlp = curve_mlp(t_mid, w0.to(device), w1.to(device))

    eval_model_mlp = fcmodel.FCModelBase(**model_config).to(device)
    offset = 0
    for param in eval_model_mlp.parameters():
        param_size = param.numel()
        param.data.copy_(w_mid_mlp[offset:offset + param_size].view(param.size()))
        offset += param_size

    acc_mlp_mid = evaluate_model(eval_model_mlp, data_loaders['test'], device)
    print(f"MLP Curve (t=0.5) Accuracy: {acc_mlp_mid:.2f}%")

    print("\n--- Training and Saving Complete ---")
    print(f"Models and curves saved in: {config['checkpoint_dir']}")
    print(f"- Model A: {model_A_path}")
    print(f"- Model B: {model_B_path}")
    print(f"- Bezier Curve: {bezier_curve_path}")
    print(f"- MLP Curve: {mlp_curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Models and Curves (Bezier & MLP) using YAML Config')

    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

    print("Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available() and config['use_cuda']:
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(config)