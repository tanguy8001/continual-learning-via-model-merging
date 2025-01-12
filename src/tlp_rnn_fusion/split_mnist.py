import pdb

import numpy as np
import os
import os.path
import torch
import warnings

from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Callable, Dict, Optional, Tuple


class SplitMNIST:
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
        nsplits: number of splits of the dataset
    """

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
            nsplits: int = 1,
            split_index: int = 1,
            scale_factor: float = 1.0
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.scale_factor = scale_factor
        assert 1 <= split_index <= nsplits

        # Downloads the dataset using the dummy creation of the MNIST torchvision dataset.
        dummy_ds = datasets.MNIST(root=root, train=train, transform=transform,
                                  target_transform=target_transform)
        self.data, self.targets = dummy_ds.data, dummy_ds.targets
        if train:
            # Create a homegeneous data split for training the models
            print("Creating nsplits:{}, index:{}".format(nsplits, split_index))
            random_seed = 543  # Same seed for same split every time
            np.random.seed(random_seed)
            random_arr = torch.from_numpy(np.random.randint(low=1,
                                                            high=nsplits + 1,
                                                            size=len(self.targets)))
            selected_index = (random_arr == split_index)
            self.data = self.data[selected_index]
            self.targets = self.targets[selected_index]
            print("Num samples:")
            for idx in range(10):
                print(self.targets[self.targets == idx].size())

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
        if self.scale_factor is not None:
            img *= self.scale_factor
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

def test_split_mnist():
    # Train for special digit
    nsplits = 10
    split_set = SplitMNIST(root='./data', nsplits=nsplits, split_index=2,
                           transform=transforms.ToTensor())
    counts = np.zeros(10)
    for i in range(0, 10):
        counts[i] = (split_set.targets == i).sum()
    print('Split set counts: ', counts)
    x, _ = split_set.__getitem__(0)
    # pdb.set_trace()


if __name__ == "__main__":
    test_split_mnist()