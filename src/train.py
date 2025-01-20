'''
The main training and fusion script. To perform a single training, simply set SEEDS to a single seed of your choice.
'''

import os
import csv
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
import ad_hoc_ot_fusion
import avg_fusion
from data import double_loaders
from models import mlpnet, fcmodel
from curve_merging import (
    train_model,
    curve_ensembling,
    CurveConfig
)
from fuse_models import get_activation_data

SEEDS = range(5)

def test_curve_merging_with_seeds():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='FC')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--result_path', type=str, default='result')

    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument("--lr", default=0.007, type=float)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        choices=['StepLR', 'MultiStepLR'])
    parser.add_argument('--lr_step_size', type=int, default=10000)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    parser.add_argument('--momentum', type=float, default=0)

    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[400,200,100])
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--rnn_steps', type=int, default=1,
                        help='Number of steps that RNN executes')
    parser.add_argument('--rnn_act_type', type=str, default='tanh',
                        choices=['tanh', 'relu'])
    parser.add_argument('--rnn_step_start', type=int, default=0,
                        help='Step number to start with for RNN experiments, helper flag')

    parser.add_argument('--log_step', type=int, default=100,
                        help='The steps after which models would be logged.')

    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)

    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument("--seed", default=24601, type=int)

    parser.add_argument('--model_path_list', type=str, default=None, nargs='+',
                        help="Comma separated list of models and checkpoints"
                             "to be used fused together")

    # Fusion parameters
    parser.add_argument('--fusion_type', type=str, default="ot",
                        choices=['tlp', 'avg', 'ot', 'fw', 'gw', 'curve'])
    parser.add_argument('--activation_batch_size', type=int, default=100)
    parser.add_argument('--use_pre_activations', default=False, action='store_true')
    parser.add_argument('--model_weights', default=None, type=float, nargs='+',
                        help='Comma separated list of weights for each model in fusion')

    parser.add_argument('--ad_hoc_cost_choice', type=str, default='activation',
                        choices=['weight', 'activation'])
    parser.add_argument('--ad_hoc_ot_solver', type=str, default='emd',
                        choices=['sinkhorn', 'emd'])
    parser.add_argument('--ad_hoc_sinkhorn_regularization', type=float, default=0)
    parser.add_argument('--ad_hoc_init_type', type=str, default=None)
    parser.add_argument('--ad_hoc_initialization', type=int, default=None)

    parser.add_argument('--theta_pi', type=float, default=1.0)
    parser.add_argument('--theta_w', type=float, default=1.0)
    parser.add_argument('--auto_optimize', type=int, default=0)

    args = parser.parse_args()
    """Main function to perform training and merging with multiple seeds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    results = {"Joint model": [], "Model A": [], "Model B": [], "OT": [], "Curve": [], "AVG": []}
    config = CurveConfig()

    data_path = os.path.join(os.getcwd(), "data")
    os.makedirs(data_path, exist_ok=True)

    for seed in SEEDS:
        print(f"\nRunning with seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data
        data_loaders, _ = double_loaders(
            dataset=config.dataset,
            path=data_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            transform_name="MLPNET",
            digit=config.test_digit,
            cifar_class=config.cifar_class,
        )
        fused_loader = create_fused_loader(
            data_loaders['trainA'],
            data_loaders['trainB'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # Initialize models
        model_A = fcmodel.FCModelBase(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.num_classes,
        ).to(device)
        print(model_A)
        model_B = fcmodel.FCModelBase(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.num_classes,
        ).to(device)
        target_model = fcmodel.FCModelBase(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.num_classes,
        ).to(device)
        joint_model = fcmodel.FCModelBase(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.num_classes,
        ).to(device)

        # Train the joint model
        print("\nTraining the joint model...")
        train_and_evaluate_model(
            model=joint_model,
            train_loader=fused_loader,
            test_loader=data_loaders['test'],
            config=config,
            model_path=f"./checkpoints/seed_{seed}/joint_model",
            device=device,
            epochs=config.model_epochs,
        )
        acc_joint = evaluate_model(joint_model, data_loaders["test"], device)
        print(f"Accuracy of the joint model: {acc_joint}")
        results["Joint model"].append(acc_joint)

        # Train and evaluate Model A
        print("\nTraining Model A...")
        train_and_evaluate_model(
            model=model_A,
            train_loader=data_loaders['trainA'],
            test_loader=data_loaders['test'],
            config=config,
            model_path=f"./checkpoints/seed_{seed}/model_A",
            device=device,
            epochs=config.model_epochs,
        )
        acc_A = evaluate_model(model_A, data_loaders['test'], device)
        print(f"Accuracy of model A: {acc_A}")
        results["Model A"].append(acc_A)

        # Train and evaluate Model B
        print("\nTraining Model B...")
        train_and_evaluate_model(
            model=model_B,
            train_loader=data_loaders['trainB'],
            test_loader=data_loaders['test'],
            config=config,
            model_path=f"./checkpoints/seed_{seed}/model_B",
            device=device,
            epochs=config.model_epochs,
        )
        acc_B = evaluate_model(model_B, data_loaders['test'], device)
        print(f"Accuracy of model B: {acc_B}")
        results["Model B"].append(acc_B)

        # Merge models
        print("\nMerging models...")
        
        print("\nOT ensembling...")
        target_model_ot = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
        ).to(device)
        OTFusionClass = ad_hoc_ot_fusion.OTFusion
        if args.ad_hoc_cost_choice == "activation":
            data = get_activation_data(args)
        else:
            data = None
        print(f"Using {args.ad_hoc_cost_choice} as the cost choice for OT.")
        fusion_method = OTFusionClass(args, base_models=[model_A, model_B],
                                           target_model=target_model_ot,
                                           data=data)
        fusion_method.fuse()
        acc_merged = evaluate_model(target_model_ot, data_loaders['test'], device)
        print(f"Accuracy of OT model: {acc_merged}")
        results["OT"].append(acc_merged)

        print("\nAVG ensembling...")
        target_model_avg = fcmodel.FCModelBase(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.num_classes,
        ).to(device)
        fusion_method = avg_fusion.AvgFusion(args, base_models=[model_A, model_B],
                                                    target_model=target_model_avg)
        fusion_method.fuse()
        acc_merged = evaluate_model(target_model_avg, data_loaders['test'], device)
        print(f"Accuracy of AVG model: {acc_merged}")
        results["AVG"].append(acc_merged)

        print("\nCurve ensembling...")
        curve_ensembling(
            config=config,
            models=[model_A, model_B],
            target_model=target_model,
            train_loader=fused_loader,
            test_loader=data_loaders['test'],
            device=device,
            num_classes=config.num_classes,
            input_dim=config.input_dim,
        )
        acc_merged = evaluate_model(target_model, data_loaders['test'], device)
        print(f"Accuracy of curve model: {acc_merged}")
        results["Curve"].append(acc_merged)

    # Compute statistics
    stats = {
        "Joint model": (np.mean(results["Joint model"]), np.std(results["Joint model"])),
        "Model A": (np.mean(results["Model A"]), np.std(results["Model A"])),
        "Model B": (np.mean(results["Model B"]), np.std(results["Model B"])),
        "Curve": (np.mean(results["Curve"]), np.std(results["Curve"])),
        "OT": (np.mean(results["OT"]), np.std(results["OT"])),
        "AVG": (np.mean(results["AVG"]), np.std(results["AVG"])),
    }

    # Save results to CSV
    save_results_to_csv(stats)

    print("\nResults:")
    for model, (mean, std) in stats.items():
        print(f"{model}: Mean Accuracy = {mean:.2f}%, Std = {std:.2f}%")

def train_and_evaluate_model(model, train_loader, test_loader, config, model_path, device, epochs):
    """
    Trains a model, evaluates it, and saves the best and final models.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing/validation data.
        config: Configuration dictionary for training.
        model_path: Directory to save the model checkpoints.
        device: The device (CPU/GPU) for training.
        epochs: Number of epochs to train.
    """
    os.makedirs(model_path, exist_ok=True)
    best_val_acc = 0
    history = []

    #print(model)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # Train the model for one epoch
        train_model(
            config,
            model,
            train_loader,
            test_loader,
            epochs=1,
            device=device,
            learning_rate=config.model_learning_rate
        )
        
        # Evaluate the model
        val_acc = evaluate_model(model, test_loader, device)
        history.append(val_acc)
        print(f"Epoch {epoch}/{epochs}, Validation Accuracy: {val_acc:.2f}%")
        
        ## Save the best model
        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    save_path = os.path.join(model_path, 'best_val_acc_model.pth')
        #    save_model(model, config, epoch, val_acc, save_path)
        #    #print(f"Best model saved at epoch {epoch} with validation accuracy {val_acc:.2f}%")
    
    # Save the final model
    final_save_path = os.path.join(model_path, 'final_model.pth')
    save_model(model, config, epochs, val_acc, final_save_path)
    print(f"Final model saved after {epochs} epochs.")
    
    return history

def save_model(model, config, epoch, val_acc, save_path):
    """
    Saves the model checkpoint.

    Args:
        model: The model to save.
        config: Configuration dictionary.
        epoch: Current epoch number.
        val_acc: Validation accuracy at the time of saving.
        save_path: Path to save the model checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'test_acc': val_acc,
        'model_state_dict': model.state_dict(),
        'config': model.get_model_config() 
    }, save_path)

class MNIST:
    """MNIST-specific transforms"""
    def __init__(self):
        self.train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test = self.train

class Transforms:
    """Container for dataset-specific transforms."""
    MNIST = MNIST()

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

def create_fused_loader(loader_A, loader_B, batch_size, num_workers):
    """Create a fused data loader from two separate loaders."""
    # Get underlying datasets
    dataset_A = loader_A.dataset
    dataset_B = loader_B.dataset
    
    # Combine datasets
    fused_dataset = ConcatDataset([dataset_A, dataset_B])
    
    # Create new loader with combined data
    fused_loader = DataLoader(
        fused_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return fused_loader

def save_results_to_csv(stats):
    """Save mean and std of accuracies to a CSV file."""
    csv_file = "model_accuracies.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Mean Accuracy (%)", "Std (%)"])
        for model, (mean, std) in stats.items():
            writer.writerow([model, mean, std])
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    test_curve_merging_with_seeds()
