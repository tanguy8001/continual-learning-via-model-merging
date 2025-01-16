import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import os

from CL.Models import *

from CL.Utils import copy_model





def train_model(model, train_loader, test_loader, epochs=10, device="cpu"):
    """Train model and return training history"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Learning rate schedule
    def get_lr(epoch):
        if epoch < epochs * 0.5:
            return 0.01
        elif epoch < epochs * 0.9:
            return 0.01 - (0.01 - 0.00007) * ((epoch - epochs * 0.5) / (epochs * 0.4))
        else:
            return 0.00007

    for epoch in range(epochs):
        # Update learning rate
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    return model


def save_model(model, path, args=None, epochs=None):
    torch.save(
        {
            "args": vars(args) if args != None else None,
            "epoch": args.n_epochs if args != None else None,
            "test_accuracy": None,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_models(seq_dataset, model, path, device="cpu"):
    while True:
        train_loader, _ = seq_dataset.get_task_data()
        current_task = seq_dataset.current_task

        current_model = copy_model(model, 784, 10)
        current_model = train_model(
            current_model, train_loader, None, epochs=10, device=device
        )
        save_model(
            current_model,
            os.path.join(path, f"model_{current_task}.checkpoint"),
        )

        if not seq_dataset.next_task():
            break