from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR100


class SequentialDataset:
    def __init__(
        self,
        root: str = "../data",
        n_tasks: int = 5,
        batch_size: int = 32,
        task_order: List[List[int]] = None,
    ):
        """
        Base class for sequential datasets.

        Args:
            root: Root directory for dataset storage
            n_tasks: Number of tasks to split dataset into
            batch_size: Batch size for dataloaders
            task_order: Optional specific ordering of class labels for tasks
        """
        self.root = root
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.current_task = 0
        self.task_order = task_order

        # These should be set by child classes
        self.train_dataset = None
        self.test_dataset = None
        self.train_targets = None
        self.test_targets = None
        self.n_classes = None

    def get_joint_data(self, batch_size=None):
        """Get data loaders for joint training (all data at once)"""
        if batch_size is None:
            batch_size = self.batch_size

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def get_task_data(self, batch_size=None) -> Tuple[DataLoader, DataLoader]:
        """Get data loaders for current task"""
        if batch_size is None:
            batch_size = self.batch_size

        if self.task_order:
            # Use custom task order
            labels = self.task_order[self.current_task]
            return self.get_task_data_with_labels(labels, batch_size)
        else:
            # Default sequential order
            classes_per_task = self.n_classes // self.n_tasks
            start_class = self.current_task * classes_per_task
            end_class = (self.current_task + 1) * classes_per_task

            train_mask = np.logical_and(
                self.train_targets >= start_class, self.train_targets < end_class
            )
            test_mask = np.logical_and(
                self.test_targets >= start_class, self.test_targets < end_class
            )

            train_data = Subset(self.train_dataset, np.where(train_mask)[0])
            test_data = Subset(self.test_dataset, np.where(test_mask)[0])

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            return train_loader, test_loader

    def get_task_data_with_labels(
        self, labels: List[int], batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Gets data for specific labels"""
        train_indices = [
            i for i, target in enumerate(self.train_targets) if target in labels
        ]
        test_indices = [
            i for i, target in enumerate(self.test_targets) if target in labels
        ]

        train_subset = Subset(self.train_dataset, train_indices)
        test_subset = Subset(self.test_dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def next_task(self) -> bool:
        """Move to next task if available"""
        if self.current_task < self.n_tasks - 1:
            self.current_task += 1
            return True
        return False


class SequentialMNIST(SequentialDataset):
    def __init__(
        self,
        root: str = "../data",
        n_tasks: int = 5,
        batch_size: int = 32,
        task_order: List[List[int]] = None,
    ):
        super().__init__(root, n_tasks, batch_size, task_order)

        # Load MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_dataset = MNIST(
            root=root, train=True, download=True, transform=transform
        )
        self.test_dataset = MNIST(
            root=root, train=False, download=True, transform=transform
        )

        # Convert targets to numpy for easier manipulation
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)
        self.n_classes = 10


def get_task_data_with_labels(seq_mnist, labels) -> Tuple[DataLoader, DataLoader]:
    """
    Gets the data for a specific task based on the provided labels.

    Args:
        seq_mnist: Your SequentialMNIST instance.
        labels: A list of labels to include in the task.

    Returns:
        train_loader: DataLoader for the training data of the task.
        test_loader: DataLoader for the test data of the task (if needed).
    """

    train_indices = [
        i
        for i, target in enumerate(seq_mnist.train_dataset.targets)
        if target in labels
    ]
    test_indices = [
        i for i, target in enumerate(seq_mnist.test_dataset.targets) if target in labels
    ]

    train_subset = Subset(seq_mnist.train_dataset, train_indices)
    test_subset = Subset(seq_mnist.test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=seq_mnist.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=seq_mnist.batch_size, shuffle=False
    )

    return train_loader, test_loader


class TaskDataset(Dataset):
    def __init__(self, dataset, mask):
        self.dataset = dataset
        self.mask = mask
        self.indices = np.where(mask)[0]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
