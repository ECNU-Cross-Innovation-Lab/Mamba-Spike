"""
Dataset loader for neuromorphic datasets using tonic library.
Supports N-MNIST, DVS Gesture, CIFAR10-DVS, and Sequential MNIST datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tonic
import tonic.transforms as transforms
from torchvision import datasets as torch_datasets
from torchvision import transforms as torch_transforms
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
            },
            'ntidigits': {
                'class': tonic.datasets.NTIDIGITS18,
                'sensor_size': (64, 1, 2),  # Cochlea: 64 frequency channels
                'num_classes': 11  # Single digits: o, 1-9, z
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

        # CIFAR10DVS doesn't have train/test split parameter
        if self.dataset_name == 'cifar10dvs':
            full_dataset = dataset_class(
                save_to=os.path.join(self.data_dir, self.dataset_name),
                transform=self.transforms
            )
            # Split manually: use first 90% for training
            train_size = int(0.9 * len(full_dataset))
            train_dataset, _ = torch.utils.data.random_split(
                full_dataset,
                [train_size, len(full_dataset) - train_size],
                generator=torch.Generator().manual_seed(42)
            )
        elif self.dataset_name == 'ntidigits':
            # NTIDIGITS needs single_digits=True for 11-class task
            train_dataset = dataset_class(
                save_to=os.path.join(self.data_dir, self.dataset_name),
                train=True,
                single_digits=True,
                transform=self.transforms
            )
        else:
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

        # CIFAR10DVS doesn't have train/test split parameter
        if self.dataset_name == 'cifar10dvs':
            full_dataset = dataset_class(
                save_to=os.path.join(self.data_dir, self.dataset_name),
                transform=self.transforms
            )
            # Split manually: use last 10% for testing
            train_size = int(0.9 * len(full_dataset))
            _, test_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, len(full_dataset) - train_size],
                generator=torch.Generator().manual_seed(42)
            )
        elif self.dataset_name == 'ntidigits':
            # NTIDIGITS needs single_digits=True for 11-class task
            test_dataset = dataset_class(
                save_to=os.path.join(self.data_dir, self.dataset_name),
                train=False,
                single_digits=True,
                transform=self.transforms
            )
        else:
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
            # CIFAR10DVS doesn't have train/test split parameter
            if self.dataset_name == 'cifar10dvs':
                _ = dataset_class(save_to=save_path)
            else:
                # This will trigger download for datasets with train/test splits
                _ = dataset_class(save_to=save_path, train=True)
                _ = dataset_class(save_to=save_path, train=False)
            print(f"Download completed!")
        else:
            print(f"{self.dataset_name} dataset already exists at {save_path}")


class SequentialMNIST(Dataset):
    """
    Sequential MNIST dataset for testing temporal modeling.
    Converts standard MNIST images to spike trains using rate coding.
    According to paper: Tests long-range temporal dependency modeling.
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        time_steps: int = 100,
        dt: float = 1.0,
        download: bool = True
    ):
        self.time_steps = time_steps
        self.dt = dt

        # Load standard MNIST
        transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])

        self.mnist = torch_datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        # image shape: (1, 28, 28), values normalized
        # Convert to spike train using rate coding
        spikes = self._image_to_spikes(image)

        # spikes shape: (time_steps, 2, 28, 28)
        # Using 2 channels to match DVS format (ON/OFF events)
        return spikes, label

    def _image_to_spikes(self, image):
        """
        Convert image to spike train using rate coding.
        Higher pixel intensity = higher spike probability
        """
        # Denormalize and get pixel intensities in [0, 1]
        pixel_values = (image + 0.1307 / 0.3081) * 0.3081
        pixel_values = torch.clamp(pixel_values, 0, 1)

        # Generate spikes over time using Poisson process
        # Higher intensity = more spikes
        spikes_list = []

        for _ in range(self.time_steps):
            # Sample spikes based on pixel intensity (rate coding)
            spike_prob = pixel_values * self.dt  # Scale by time step
            spikes_on = torch.bernoulli(spike_prob)  # ON events

            # Create OFF events as inverse (for contrast)
            spikes_off = torch.bernoulli((1 - pixel_values) * self.dt * 0.5)

            # Stack ON and OFF channels
            spikes_t = torch.cat([spikes_on, spikes_off], dim=0)  # (2, 28, 28)
            spikes_list.append(spikes_t)

        # Stack time steps: (time_steps, 2, 28, 28)
        spikes = torch.stack(spikes_list, dim=0)

        return spikes.float()


def prepare_sequential_mnist_dataset(
    data_dir: str = "./data",
    batch_size: int = 32,
    time_steps: int = 100,
    dt: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare Sequential MNIST dataset for testing temporal modeling.

    Args:
        data_dir: Directory to save/load data
        batch_size: Batch size for training
        time_steps: Number of time steps for spike encoding
        dt: Time resolution for spike encoding
        num_workers: Number of data loading workers

    Returns:
        train_loader, test_loader, num_classes
    """
    # Create datasets
    train_dataset = SequentialMNIST(
        root=data_dir,
        train=True,
        time_steps=time_steps,
        dt=dt,
        download=True
    )

    test_dataset = SequentialMNIST(
        root=data_dir,
        train=False,
        time_steps=time_steps,
        dt=dt,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, 10  # MNIST has 10 classes


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


def prepare_ntidigits_dataset(
    data_dir: str = "./data",
    batch_size: int = 32,
    time_window: int = 300000,  # 300ms for audio
    dt: float = 1000,  # 1ms bins
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Prepare N-TIDIGITS dataset for audio recognition.

    N-TIDIGITS is a neuromorphic audio dataset created using a cochlea model.
    Uses single digit recognition task (11 classes: o, 1-9, z).

    Args:
        data_dir: Directory to save/load data
        batch_size: Batch size for training
        time_window: Time window for audio events (microseconds)
        dt: Time bin resolution (microseconds)
        num_workers: Number of data loading workers

    Returns:
        train_loader, test_loader, num_classes
    """
    dataset = NeuromorphicDataset(
        dataset_name='ntidigits',
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