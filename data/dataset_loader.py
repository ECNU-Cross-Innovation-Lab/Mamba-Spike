"""
Dataset loader for neuromorphic datasets using tonic library.
Supports N-MNIST, DVS Gesture, CIFAR10-DVS datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from typing import Optional, Tuple, Dict


class NeuromorphicDataset:
    """Unified loader for neuromorphic datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "./data",
        time_window: int = 10000,  # microseconds
        sensor_size: Optional[Tuple[int, int]] = None,
        dt: float = 1000,  # time bin in microseconds
        transform: Optional[object] = None
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.time_window = time_window
        self.dt = dt
        self.transform = transform
        
        # Dataset specific configurations
        self.dataset_configs = {
            'nmnist': {
                'class': tonic.datasets.NMNIST,
                'sensor_size': (34, 34, 2),
                'num_classes': 10
            },
            'dvsgesture': {
                'class': tonic.datasets.DVSGesture,
                'sensor_size': (128, 128, 2),
                'num_classes': 11
            },
            'cifar10dvs': {
                'class': tonic.datasets.CIFAR10DVS,
                'sensor_size': (128, 128, 2),
                'num_classes': 10
            }
        }
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        self.sensor_size = sensor_size or config['sensor_size']  # Keep full 3-tuple
        self.num_classes = config['num_classes']
        
        # Create transforms
        self.transforms = self._create_transforms()
        
    def _create_transforms(self):
        """Create preprocessing transforms for event data."""
        transform_list = []
        
        # Denoise by removing isolated events
        transform_list.append(
            transforms.Denoise(filter_time=10000)
        )
        
        # Convert to frames
        # Use n_time_bins to specify the number of time bins
        transform_list.append(
            transforms.ToFrame(
                sensor_size=self.sensor_size,
                n_time_bins=int(self.time_window / self.dt)
            )
        )
        
        return transforms.Compose(transform_list)
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4):
        """Get training data loader."""
        dataset_class = self.dataset_configs[self.dataset_name]['class']
        
        train_dataset = dataset_class(
            save_to=os.path.join(self.data_dir, self.dataset_name),
            train=True,
            transform=self.transforms
        )
        
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._custom_collate_fn
        )
    
    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False, num_workers: int = 4):
        """Get test data loader."""
        dataset_class = self.dataset_configs[self.dataset_name]['class']
        
        test_dataset = dataset_class(
            save_to=os.path.join(self.data_dir, self.dataset_name),
            train=False,
            transform=self.transforms
        )
        
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._custom_collate_fn
        )
    
    def _custom_collate_fn(self, batch):
        """Custom collate function to handle event data."""
        frames_list = []
        labels_list = []
        
        for frames, label in batch:
            # frames shape: [time_bins, channels, height, width]
            frames_list.append(torch.from_numpy(frames).float())
            labels_list.append(label)
        
        # Stack into batch
        frames_batch = torch.stack(frames_list)  # [batch, time, channels, height, width]
        labels_batch = torch.tensor(labels_list)
        
        return frames_batch, labels_batch
    
    def download_dataset(self):
        """Download dataset if not already present."""
        dataset_class = self.dataset_configs[self.dataset_name]['class']
        save_path = os.path.join(self.data_dir, self.dataset_name)
        
        if not os.path.exists(save_path):
            print(f"Downloading {self.dataset_name} dataset...")
            # This will trigger download
            _ = dataset_class(save_to=save_path, train=True)
            _ = dataset_class(save_to=save_path, train=False)
            print(f"Download completed!")
        else:
            print(f"{self.dataset_name} dataset already exists at {save_path}")


def prepare_nmnist_dataset(
    data_dir: str = "./data",
    batch_size: int = 32,
    time_window: int = 300000,  # 300ms
    dt: float = 1000,  # 1ms bins
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare N-MNIST dataset for training.
    
    Returns:
        train_loader, test_loader, num_classes
    """
    dataset = NeuromorphicDataset(
        dataset_name='nmnist',
        data_dir=data_dir,
        time_window=time_window,
        dt=dt
    )
    
    # Download if necessary
    dataset.download_dataset()
    
    # Get data loaders
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
    data_dir: str = "./data",
    batch_size: int = 16,
    time_window: int = 500000,  # 500ms
    dt: float = 1000,  # 1ms bins
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare DVS Gesture dataset."""
    dataset = NeuromorphicDataset(
        dataset_name='dvsgesture',
        data_dir=data_dir,
        time_window=time_window,
        dt=dt
    )
    
    dataset.download_dataset()
    
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
    data_dir: str = "./data",
    batch_size: int = 32,
    time_window: int = 1000000,  # 1s
    dt: float = 10000,  # 10ms bins
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare CIFAR10-DVS dataset."""
    dataset = NeuromorphicDataset(
        dataset_name='cifar10dvs',
        data_dir=data_dir,
        time_window=time_window,
        dt=dt
    )
    
    dataset.download_dataset()
    
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
    # Test dataset loading
    print("Testing N-MNIST dataset loading...")
    train_loader, test_loader, num_classes = prepare_nmnist_dataset(
        batch_size=4,
        num_workers=0  # For testing
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get one batch
    data, labels = next(iter(train_loader))
    print(f"Data shape: {data.shape}")  # [batch, time, channels, height, width]
    print(f"Labels shape: {labels.shape}")