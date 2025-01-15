import numpy as np
import os
import os.path
import torch
import warnings

from PIL import Image
from torchvision import datasets
from typing import Any, Callable, Dict, Optional, Tuple


class HeteroMNIST:
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset. Custom version here.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            special_digit: int = 4,
            other_digits_train_split: float = 0.90,
            special_train_split: bool = True
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # Downloads the dataset using the dummy creation of the MNIST torchvision dataset.
        dummy_ds = datasets.MNIST(root=root, train=train, transform=transform,
                                  target_transform=target_transform)
        self.data, self.targets = dummy_ds.data, dummy_ds.targets

        if train:
            # Create a heterogenous data split for training the models
            random_seed = 543  # Same seed for same split every time
            np.random.seed(random_seed)
            random_arr = torch.from_numpy(np.random.rand(len(self.targets)))
            special_digit_index = self.targets == special_digit
            if special_train_split:
                selected_index = special_digit_index | (random_arr >= other_digits_train_split)
            else:
                selected_index = (~special_digit_index) & (random_arr < other_digits_train_split)
            self.data = self.data[selected_index]
            self.targets = self.targets[selected_index]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')


########## TESTS ############

def test_hetero_mnist():
    # Train for special digit
    special_set = HeteroMNIST(root='./data', special_train_split=True, special_digit=4)
    other_set = HeteroMNIST(root='./data', special_train_split=False, special_digit=4)
    special_counts = np.zeros(10)
    other_counts = np.zeros(10)
    for i in range(0, 10):
        special_counts[i] = (special_set.targets == i).sum()
        other_counts[i] = (other_set.targets == i).sum()
    print('Special set counts: ', special_counts)
    print('Other set counts: ', other_counts)
    assert other_counts[4] == 0


if __name__ == "__main__":
    test_hetero_mnist()
