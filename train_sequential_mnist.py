#!/usr/bin/env python3
"""
Sequential MNIST Dataset Training Script
Target Accuracy: 99.4% (from paper)
Auto-stops when test accuracy reaches 99% of target (98.406%)
"""

import os
import sys
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset_loader import prepare_sequential_mnist_dataset
from models.mamba_spike import create_mamba_spike_sequential_mnist


class SequentialMNISTTrainer:
    def __init__(self, batch_size=32, lr=0.001, max_epochs=200):
        # Paper target and early stopping threshold
        self.target_accuracy = 99.4
        self.early_stop_threshold = self.target_accuracy * 0.99  # 98.406%

        # Setup device
        self.device = self._setup_device()

        # Training config
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs

        # Create output directory
        self.output_dir = os.path.join(
            "results",
            f"sequential_mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Save config
        self.config = {
            "dataset": "sequential_mnist",
            "target_accuracy": self.target_accuracy,
            "early_stop_threshold": self.early_stop_threshold,
            "batch_size": batch_size,
            "lr": lr,
            "max_epochs": max_epochs,
            "device": str(self.device)
        }

        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Load dataset
        print("\n" + "="*70)
        print("Sequential MNIST Training")
        print("="*70)
        print(f"Target Accuracy: {self.target_accuracy}%")
        print(f"Early Stop Threshold: {self.early_stop_threshold:.2f}%")
        print(f"Max Epochs: {max_epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")

        print("Loading Sequential MNIST dataset...")
        self.train_loader, self.test_loader, self.num_classes = prepare_sequential_mnist_dataset(
            batch_size=batch_size,
            num_workers=0,  # Avoid DataLoader issues
            num_steps=100,
            dt=1
        )
        print(f"✓ Dataset loaded: {len(self.train_loader)} train batches, "
              f"{len(self.test_loader)} test batches\n")

        # Create model
        print("Creating model...")
        self.model = create_mamba_spike_sequential_mnist().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model created: {total_params:,} parameters\n")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.0001
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=lr * 0.01
        )

        # Tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.start_time = time.time()

    def _setup_device(self):
        """Setup computing device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                _ = torch.zeros(1).to(device)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"\n✓ GPU detected: {gpu_name}")
                return device
            except:
                print("\n⚠ CUDA available but initialization failed, using CPU")
                return torch.device("cpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            try:
                _ = torch.zeros(1).to(device)
                print(f"\n✓ Apple Silicon GPU (MPS) detected")
                return device
            except:
                print("\n⚠ MPS available but initialization failed, using CPU")
                return torch.device("cpu")
        else:
            print("\n⚠ No GPU detected, using CPU")
            return torch.device("cpu")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total

        return train_loss, train_acc

    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        test_loss = total_loss / len(self.test_loader)
        test_acc = 100. * correct / total

        return test_loss, test_acc

    def save_checkpoint(self, epoch, test_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'test_acc': test_acc,
            'best_acc': self.best_acc,
            'config': self.config
        }

        # Always save latest
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model: {test_acc:.2f}%")

    def train(self):
        """Main training loop with early stopping."""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70 + "\n")

        try:
            for epoch in range(1, self.max_epochs + 1):
                # Train
                train_loss, train_acc = self.train_epoch(epoch)

                # Evaluate
                test_loss, test_acc = self.evaluate()

                # Update scheduler
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log to TensorBoard
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/acc', train_acc, epoch)
                self.writer.add_scalar('test/loss', test_loss, epoch)
                self.writer.add_scalar('test/acc', test_acc, epoch)
                self.writer.add_scalar('train/lr', current_lr, epoch)

                # Print results
                print(f"\nEpoch {epoch}/{self.max_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")

                # Check if best
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self.best_epoch = epoch

                # Save checkpoint
                self.save_checkpoint(epoch, test_acc, is_best)

                # Check early stopping
                if test_acc >= self.early_stop_threshold:
                    elapsed = time.time() - self.start_time
                    print("\n" + "="*70)
                    print("🎯 TARGET REACHED!")
                    print("="*70)
                    print(f"Target: {self.target_accuracy}%")
                    print(f"Threshold: {self.early_stop_threshold:.2f}%")
                    print(f"Achieved: {test_acc:.2f}%")
                    print(f"Epoch: {epoch}/{self.max_epochs}")
                    print(f"Training time: {elapsed/3600:.2f} hours")
                    print("="*70 + "\n")

                    # Save final results
                    results = {
                        'target_reached': True,
                        'target_accuracy': self.target_accuracy,
                        'threshold': self.early_stop_threshold,
                        'achieved_accuracy': test_acc,
                        'best_accuracy': self.best_acc,
                        'final_epoch': epoch,
                        'total_epochs': self.max_epochs,
                        'training_time_hours': elapsed / 3600
                    }

                    with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
                        json.dump(results, f, indent=2)

                    print(f"✓ Results saved to: {self.output_dir}")
                    break

                # Progress update
                print(f"  Best so far: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
                print(f"  Target: {self.early_stop_threshold:.2f}% "
                      f"(Gap: {self.early_stop_threshold - test_acc:.2f}%)")
                print()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")

        finally:
            # Final summary
            elapsed = time.time() - self.start_time
            print("\n" + "="*70)
            print("Training Complete")
            print("="*70)
            print(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
            print(f"Target Accuracy: {self.target_accuracy}%")
            print(f"Gap to target: {self.target_accuracy - self.best_acc:.2f}%")
            print(f"Total training time: {elapsed/3600:.2f} hours")
            print(f"Results saved to: {self.output_dir}")
            print("="*70 + "\n")

            self.writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train on Sequential MNIST dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum epochs (default: 200)')
    args = parser.parse_args()

    trainer = SequentialMNISTTrainer(
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs
    )

    trainer.train()


if __name__ == "__main__":
    main()
