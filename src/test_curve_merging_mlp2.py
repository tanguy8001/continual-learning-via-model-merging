"""
Trains two models on CIFAR10/MNIST dataset and merges them using curve ensembling with MLP.
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb # Add wandb import

from data import double_loaders, create_fused_loader

from models import mlpnet, fcmodel
from curve_merging import (
    train_model,
    curve_ensembling,
    CurveConfig,
    evaluate_model,
)
# Remove CurveMLP import
# from curves_MLP import CurveMLP
# Import CurveNet and LearnedCoeffLayerMLP
from curves import CurveNet, LearnedCoeffLayerMLP, l2_regularizer

def main():

    config = CurveConfig(
        dataset="CIFAR10",
        input_dim=3072,  # 32x32x3
        batch_size=128,
        model_epochs=10, # epochs to train base models A and B
        epochs=10,  # epochs for CurveNet training
        num_bends=3, # Number of control points (start, middle, end)
        learning_rate=0.07, # LR for CurveNet training
        momentum=0.9,
        weight_decay=1e-4 # WD for CurveNet training
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create data directory if it doesn't exist
    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)

    data_loaders, num_classes = double_loaders(
        dataset=config.dataset,
        path=data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform_name=config.transform,
        digit=config.test_digit,
        cifar_class=config.cifar_class,
        #output_dim=config.num_classes, # Pass num_classes
    )

    # Create fused loader for curve training
    fused_loader = create_fused_loader(
        data_loaders['trainA'],
        data_loaders['trainB'],
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # Initialize base models A and B
    base_model_kwargs = {
        'input_dim': config.input_dim,
        'hidden_dims': config.hidden_dims,
        'output_dim': num_classes,
        'bias': False # <<< Change bias to False to match saved checkpoints
    }
    # It might be cleaner to use FCModel.base directly
    model_A = fcmodel.FCModel.base(**base_model_kwargs).to(device)
    model_B = fcmodel.FCModel.base(**base_model_kwargs).to(device)

    # Check if models exist and load them if they do
    model_A_path = f"checkpoints/{config.dataset.lower()}_model_A.pth"
    model_B_path = f"checkpoints/{config.dataset.lower()}_model_B.pth"

    if os.path.exists(model_A_path) and os.path.exists(model_B_path):
        print("Loading pre-trained models...")
        model_A.load_state_dict(torch.load(model_A_path, map_location=device))
        model_B.load_state_dict(torch.load(model_B_path, map_location=device))
    else:
        print("Training Model A...")
        train_model(config, model_A, data_loaders['trainA'], data_loaders['test'], epochs=config.model_epochs, device=device)
        print("Training Model B...")
        train_model(config, model_B, data_loaders['trainB'], data_loaders['test'], epochs=config.model_epochs, device=device)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model_A.state_dict(), model_A_path)
        torch.save(model_B.state_dict(), model_B_path)
        print("Models saved to checkpoints directory")

    # Evaluate individual models
    acc_A = evaluate_model(model_A, data_loaders['test'], device)
    acc_B = evaluate_model(model_B, data_loaders['test'], device)

    print(f"Model A accuracy: {acc_A:.2f}%")
    print(f"Model B accuracy: {acc_B:.2f}%")

    # --- CurveNet setup ---
    print("Initializing CurveNet with LearnedCoeffLayerMLP...")

    # Pass FCModelCurve as the architecture to CurveNet
    # Adjust architecture_kwargs to match FCModelCurve's expected args (if different)
    # FCModelCurve expects: input_dim, hidden_dims, output_dim, fix_points, bias
    # fix_points is handled by CurveNet, others are in base_model_kwargs
    curve_model_kwargs = base_model_kwargs.copy() # Start with base args (now has bias=False)
    # Remove kwargs not expected by FCModelCurve or handled by CurveNet
    # (In this case, they seem to match, but good practice to check)

    curve_model = CurveNet(
        num_classes=num_classes,
        curve=LearnedCoeffLayerMLP, # Use the MLP coefficient layer
        architecture=fcmodel.FCModel.curve, # <<< Use FCModelCurve here
        num_bends=config.num_bends,
        fix_start=True,
        fix_end=True,
        architecture_kwargs=curve_model_kwargs # Pass appropriate kwargs (with bias=False)
    ).to(device)

    # Load weights of model_A and model_B into the fixed start and end points
    print("Loading base models into CurveNet endpoints...")
    curve_model.import_base_parameters(model_A, 0)
    curve_model.import_base_parameters(model_B, config.num_bends - 1)
    curve_model.import_base_buffers(model_A)

    # --- Training the CurveNet ---
    print("Training the CurveNet...")
    criterion = nn.CrossEntropyLoss()
    regularizer = l2_regularizer(config.weight_decay) if config.weight_decay > 0 else None

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, curve_model.parameters()),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # --- Wandb setup ---
    run = wandb.init(
        entity="Continual_Learning-DAL",
        project="Model Path Fusion - CurveNet MLP Coeffs",
        config={
            "device": device,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "optimizer": "SGD",
            "weight_decay": config.weight_decay,
            "num_bends": config.num_bends,
            "dataset": config.dataset,
            "architecture": curve_model.architecture.__name__, # This will now be FCModelCurve
            "coeff_layer": curve_model.curve.__name__,
            "bias": curve_model_kwargs.get('bias', False) # Log bias if used
        }
    )

    for epoch in range(config.epochs):
        curve_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        train_iterator = tqdm(fused_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

        for batch_idx, (inputs, targets) in enumerate(train_iterator):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Sample t uniformly from [0, 1] and keep it as a tensor
            # t = torch.rand(1, device=device).item() # Old: converts to float
            t_tensor = torch.rand(1, device=device) # New: keep as tensor

            # Forward pass through CurveNet
            outputs = curve_model(inputs, t=t_tensor)
            loss = criterion(outputs, targets)

            if regularizer is not None:
                loss += regularizer(curve_model)

            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            total_loss += loss.item()

            train_iterator.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(fused_loader)
        accuracy = 100. * total_correct / total_samples
        print(f'Epoch: {epoch+1}/{config.epochs}, Average Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')

        # Evaluate on test set at midpoint t=0.5
        curve_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_samples = 0
        with torch.no_grad():
            for inputs, targets in data_loaders['test']:
                inputs, targets = inputs.to(device), targets.to(device)
                t_eval_tensor = torch.tensor([0.5], device=device) # Use tensor for eval t
                outputs = curve_model(inputs, t=t_eval_tensor) # Evaluate at midpoint
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_samples += targets.size(0)

        avg_test_loss = test_loss / len(data_loaders['test'])
        test_accuracy = 100. * test_correct / test_samples
        print(f'Epoch: {epoch+1}/{config.epochs}, Test Loss (t=0.5): {avg_test_loss:.4f}, Test Accuracy (t=0.5): {test_accuracy:.2f}%')

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "test_loss_t0.5": avg_test_loss,
            "test_accuracy_t0.5": test_accuracy
        })

    # --- Evaluation after training (Optional: evaluate at multiple t) ---
    print("\nEvaluating final CurveNet model...")
    final_accuracies = {}
    curve_model.eval()
    with torch.no_grad():
        for t_eval_float in [0.0, 0.25, 0.5, 0.75, 1.0]:
             t_eval_tensor = torch.tensor([t_eval_float], device=device) # Use tensor for eval t
             test_correct = 0
             test_samples = 0
             for inputs, targets in data_loaders['test']:
                 inputs, targets = inputs.to(device), targets.to(device)
                 outputs = curve_model(inputs, t=t_eval_tensor)
                 _, predicted = outputs.max(1)
                 test_correct += predicted.eq(targets).sum().item()
                 test_samples += targets.size(0)
             acc = 100. * test_correct / test_samples
             final_accuracies[f't={t_eval_float}'] = acc
             print(f"Accuracy at t={t_eval_float:.2f}: {acc:.2f}%")

    wandb.log({"final_accuracies": final_accuracies})
    run.finish() # End wandb run

    # --- Save the trained CurveNet model ---
    curve_model_path = f"checkpoints/{config.dataset.lower()}_curve_mlpcoeff_net.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(curve_model.state_dict(), curve_model_path)
    print(f"Trained CurveNet model saved to {curve_model_path}")

    # --- Cleanup (Optional: Delete previous CurveMLP checkpoint if desired) ---
    # old_curve_mlp_path = f"checkpoints/{config.dataset.lower()}_curve_mlp.pth"
    # if os.path.exists(old_curve_mlp_path):
    #     os.remove(old_curve_mlp_path)
    #     print(f"Removed old checkpoint: {old_curve_mlp_path}")


if __name__ == "__main__":
    main()
