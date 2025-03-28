import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List
import logging


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Evaluate model on data_loader"""
    model = model.eval()
    total_loss = 0
    batch_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            batch_correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * batch_correct / total
    return avg_loss, accuracy


def evaluate_full_dataset(
    model, dataset, batch_size=32, device="cpu"
) -> tuple[float, float]:
    """Evaluate model on the full dataset"""
    # Create a dataloader for the full dataset
    test_set = dataset.test_dataset
    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    avg_loss, accuracy = evaluate_model(model, data_loader, criterion, device)
    return avg_loss, accuracy


def evaluate_task_accuracies(model, seq_data, n_tasks=5, batch_size=32, device="cpu"):
    """Evaluate model accuracy on each task separately"""

    task_accuracies = []

    seq_data.current_task = 0

    total_accuracy = 0

    for task in range(seq_data.n_tasks):
        _, test_loader = seq_data.get_task_data()
        _, accuracy = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
        task_accuracies.append(accuracy)
        seq_data.next_task()

        total_accuracy += (
            accuracy * len(test_loader.dataset) / len(seq_data.test_dataset)
        )

    return task_accuracies, total_accuracy


def evaluate_task_accuracies_custom_order(
    model: torch.nn.Module,
    test_dataset,
    task_order: List[List[int]],
    device: str = "cpu",
) -> List[float]:
    """
    Evaluates the model's accuracy on each task defined by the custom task order.

    Args:
        model: The trained model.
        test_dataset: The test dataset.
        task_order: The custom task order.
        device: The device to use.

    Returns:
        List: A list of accuracies, one for each task.
    """
    accuracies = []
    for task_labels in task_order:
        test_indices = [
            i for i, target in enumerate(test_dataset.targets) if target in task_labels
        ]
        if not test_indices:
            accuracies.append(0.0)
            logging.warning(f"No test data found for task with labels: {task_labels}")
            continue

        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0.0
        accuracies.append(accuracy)

    return accuracies
