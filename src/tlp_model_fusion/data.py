import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class Transforms:

    class MNIST:

        class MLPNET:

            # For MNIST, no need for color normalization as images are grayscale
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])  # mean and std for MNIST grayscale
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])  # mean and std for MNIST grayscale
            ])

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    CIFAR100 = CIFAR10


def loaders(dataset, path, batch_size, num_workers, transform_name, exclude_digit=None, only_digit=None, use_test=False,
            shuffle_train=True):
    # Determine the dataset class from torchvision
    ds = getattr(datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    
    # Load the dataset (MNIST or CIFAR10)
    if dataset.lower() == "mnist":
        print("Loaded MNIST.")
        train_set = datasets.MNIST(root=path, train=True, download=True, transform=transform.train)
        test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform.test)
        
        # Exclude the specified digit from the training set
        if exclude_digit is not None:
            print(f'Excluding digit {exclude_digit} for training.')
            train_indices = [i for i in range(len(train_set)) if train_set[i][1] != exclude_digit]
            #test_indices = [i for i in range(len(test_set)) if test_set[i][1] != exclude_digit]
            train_set = Subset(train_set, train_indices)
            #test_set = Subset(test_set, test_indices)  # You can choose to exclude from test as well

        elif only_digit is not None:
            print(f'Training ONLY on digit {only_digit}.')
            only_digit_indices = [i for i, (x, y) in enumerate(train_set) if y == only_digit]
            train_set = Subset(train_set, only_digit_indices)

    else:
        print("Loaded CIFAR10")
        train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform.train)
        test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform.test)

    # Access the targets from the underlying dataset if it's a Subset
    if isinstance(train_set, Subset):
        train_targets = train_set.dataset.targets
    else:
        train_targets = train_set.targets

    # Return the DataLoaders for training and testing
    return {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
    }, max(train_targets) + 1  # Use train_targets which is accessible from dataset


def double_loaders(dataset, path, batch_size, num_workers, transform_name, digit=None, use_test=False,
            shuffle_train=True):
    # Determine the dataset class from torchvision
    ds = getattr(datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)
    
    # Load the dataset (MNIST or CIFAR10)
    if dataset.lower() == "mnist":
        print("Loaded MNIST.")
        train_set = datasets.MNIST(root=path, train=True, download=True, transform=transform.train)
        test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform.test)
        
        if digit is not None:
            print(f'Using the heterogeneous data-split setup with digit {digit}.')
            
            # Separate indices for digit 4 and other digits
            indices_digit = [i for i, (x, y) in enumerate(train_set) if y == digit]
            other_digits_indices = [i for i, (x, y) in enumerate(train_set) if y != digit]

            # Split data for models A and B
            # Model A gets 10% of data for digit 'only_digit'
            indices_10_percent = other_digits_indices[:int(0.1 * len(other_digits_indices))]

            # Model B gets 90% of the training data for remaining digits
            indices_90_percent = other_digits_indices[int(0.1 * len(other_digits_indices)):]

            # Create data subsets
            dataset_A = Subset(train_set, indices_10_percent)
            dataset_A += Subset(train_set, indices_digit)
            dataset_B = Subset(train_set, indices_90_percent)

    # Access the targets from the underlying dataset if it's a Subset
    if isinstance(train_set, Subset):
        train_targets = train_set.dataset.targets
    else:
        train_targets = train_set.targets


    return {
        'trainA': torch.utils.data.DataLoader(
            dataset_A,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        ),
        'trainB': torch.utils.data.DataLoader(
            dataset_B,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
    }, max(train_targets) + 1  # Use train_targets which is accessible from dataset