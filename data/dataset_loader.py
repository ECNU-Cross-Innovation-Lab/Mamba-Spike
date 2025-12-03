"""
Dataset loader for neuromorphic datasets.
Loads preprocessed data from local pickle files.
Supports N-MNIST, DVS Gesture, CIFAR10-DVS, Sequential MNIST, and N-TIDIGITS datasets.
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple


class PreprocessedDataset(Dataset):
    """Dataset class for loading preprocessed neuromorphic data from pickle files."""

    def __init__(self, data_path: str):
        """
        Initialize dataset from a pickle file.

        Args:
            data_path: Path to the pickle file containing preprocessed data.
                      Expected format: {'data': np.ndarray, 'labels': np.ndarray}
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        self.data = dataset['data']
        self.labels = dataset['labels']

        assert len(self.data) == len(self.labels), "Data and labels must have same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx]).float()
        label = int(self.labels[idx])
        return data, label


class NeuromorphicDataset:
    """Unified loader for neuromorphic datasets from preprocessed pickle files."""

    DATASET_CONFIGS = {
        'nmnist': {
            'sensor_size': (34, 34, 2),
            'num_classes': 10
        },
        'dvsgesture': {
            'sensor_size': (128, 128, 2),
            'num_classes': 11
        },
        'cifar10dvs': {
            'sensor_size': (128, 128, 2),
            'num_classes': 10
        },
        'ntidigits': {
            'sensor_size': (64, 1, 2),
            'num_classes': 11
        },
        'sequential_mnist': {
            'sensor_size': (28, 28, 2),
            'num_classes': 10
        }
    }

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data/preprocessed"
    ):
        """
        Initialize neuromorphic dataset loader.

        Args:
            dataset_name: Name of the dataset ('nmnist', 'dvsgesture', 'cifar10dvs',
                         'sequential_mnist', 'ntidigits')
            data_dir: Directory containing preprocessed pickle files
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir

        if self.dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Choose from: {list(self.DATASET_CONFIGS.keys())}"
            )

        config = self.DATASET_CONFIGS[self.dataset_name]
        self.sensor_size = config['sensor_size']
        self.num_classes = config['num_classes']

        self.train_path = os.path.join(data_dir, f"{self.dataset_name}_train.pkl")
        self.test_path = os.path.join(data_dir, f"{self.dataset_name}_test.pkl")

    def get_train_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Get training data loader."""
        train_dataset = PreprocessedDataset(self.train_path)

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_test_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 4
    ) -> DataLoader:
        """Get test data loader."""
        test_dataset = PreprocessedDataset(self.test_path)

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def prepare_nmnist_dataset(
    data_dir: str = "./data/preprocessed",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare N-MNIST dataset for training.

    Returns:
        train_loader, test_loader, num_classes
    """
    dataset = NeuromorphicDataset(
        dataset_name='nmnist',
        data_dir=data_dir
    )

    train_loader = dataset.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, dataset.num_classes


def prepare_dvsgesture_dataset(
    data_dir: str = "./data/preprocessed",
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare DVS Gesture dataset."""
    dataset = NeuromorphicDataset(
        dataset_name='dvsgesture',
        data_dir=data_dir
    )

    train_loader = dataset.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, dataset.num_classes


def prepare_cifar10dvs_dataset(
    data_dir: str = "./data/preprocessed",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare CIFAR10-DVS dataset."""
    dataset = NeuromorphicDataset(
        dataset_name='cifar10dvs',
        data_dir=data_dir
    )

    train_loader = dataset.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, dataset.num_classes


def prepare_sequential_mnist_dataset(
    data_dir: str = "./data/preprocessed",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare Sequential MNIST dataset for testing temporal modeling.

    Returns:
        train_loader, test_loader, num_classes
    """
    dataset = NeuromorphicDataset(
        dataset_name='sequential_mnist',
        data_dir=data_dir
    )

    train_loader = dataset.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, dataset.num_classes


def prepare_ntidigits_dataset(
    data_dir: str = "./data/preprocessed",
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare N-TIDIGITS dataset for audio recognition.

    Returns:
        train_loader, test_loader, num_classes
    """
    dataset = NeuromorphicDataset(
        dataset_name='ntidigits',
        data_dir=data_dir
    )

    train_loader = dataset.get_train_loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = dataset.get_test_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, dataset.num_classes


if __name__ == "__main__":
    print("Testing N-MNIST dataset loading...")
    try:
        train_loader, test_loader, num_classes = prepare_nmnist_dataset(
            batch_size=4,
            num_workers=0
        )

        print(f"Number of classes: {num_classes}")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of test batches: {len(test_loader)}")

        data, labels = next(iter(train_loader))
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please ensure preprocessed data files are in ./data/preprocessed/")
