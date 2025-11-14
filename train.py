"""
Training script for Mamba-Spike model on neuromorphic datasets.
"""

import os
import argparse
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset_loader import (
    prepare_nmnist_dataset,
    prepare_dvsgesture_dataset,
    prepare_cifar10dvs_dataset,
    prepare_sequential_mnist_dataset
)
from models.mamba_spike import (
    create_mamba_spike_nmnist,
    create_mamba_spike_dvsgesture,
    create_mamba_spike_cifar10dvs,
    create_mamba_spike_sequential_mnist
)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = os.path.join(
            args.output_dir, 
            f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # Load dataset
        self.train_loader, self.test_loader, self.num_classes = self._load_dataset()
        
        # Create model
        self.model = self._create_model().to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # Best accuracy tracking
        self.best_acc = 0.0

    def _setup_device(self):
        """
        Automatically detect and setup the best available device.
        Priority: CUDA > MPS > CPU
        """
        # Try CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                # Test CUDA initialization
                _ = torch.zeros(1).to(device)
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"\nGPU detected: {gpu_name}")
                print(f"GPU memory: {gpu_memory:.1f} GB")
                print(f"CUDA version: {torch.version.cuda}")
                return device
            except Exception as e:
                print(f"\nWarning: CUDA available but initialization failed: {e}")
                print("Falling back to CPU...")

        # Try MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            try:
                # Test MPS initialization
                _ = torch.zeros(1).to(device)
                print(f"\nApple Silicon GPU (MPS) detected")
                return device
            except Exception as e:
                print(f"\nWarning: MPS available but initialization failed: {e}")
                print("Falling back to CPU...")

        # Fallback to CPU
        print("\nNo GPU detected, using CPU")
        print("Note: Training will be slower on CPU. Consider using a GPU for better performance.")
        return torch.device("cpu")

    def _load_dataset(self):
        """Load the specified dataset."""
        print(f"Loading {self.args.dataset} dataset...")
        
        if self.args.dataset == 'nmnist':
            return prepare_nmnist_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        elif self.args.dataset == 'dvsgesture':
            return prepare_dvsgesture_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        elif self.args.dataset == 'cifar10dvs':
            return prepare_cifar10dvs_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        elif self.args.dataset == 'sequential_mnist':
            return prepare_sequential_mnist_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_steps=int(self.args.time_window / self.args.dt),
                dt=self.args.dt / 1000.0,  # Convert to probability scale
                num_workers=self.args.num_workers
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def _create_model(self):
        """Create the model for the specified dataset."""
        if self.args.dataset == 'nmnist':
            return create_mamba_spike_nmnist(num_classes=self.num_classes)
        elif self.args.dataset == 'dvsgesture':
            return create_mamba_spike_dvsgesture(num_classes=self.num_classes)
        elif self.args.dataset == 'cifar10dvs':
            return create_mamba_spike_cifar10dvs(num_classes=self.num_classes)
        elif self.args.dataset == 'sequential_mnist':
            return create_mamba_spike_sequential_mnist(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.args.clip_grad
                )
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar('train/acc', 100. * correct / total, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, epoch):
        """Evaluate on test set."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        # Log to tensorboard
        self.writer.add_scalar('test/loss', avg_loss, epoch)
        self.writer.add_scalar('test/acc', accuracy, epoch)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'args': self.args
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_latest.pth'))
        
        # Save best checkpoint
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            torch.save(checkpoint, os.path.join(self.output_dir, 'checkpoint_best.pth'))
            print(f"New best accuracy: {accuracy:.2f}%")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate
            test_loss, test_acc = self.evaluate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('train/lr', current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, test_acc)
        
        print(f"\nTraining completed! Best accuracy: {self.best_acc:.2f}%")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Mamba-Spike model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='nmnist',
                        choices=['nmnist', 'dvsgesture', 'cifar10dvs', 'sequential_mnist'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save datasets')
    
    # Model
    parser.add_argument('--time-window', type=int, default=300000,
                        help='Time window in microseconds')
    parser.add_argument('--dt', type=float, default=1000,
                        help='Time bin in microseconds')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()