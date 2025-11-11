"""
Evaluation script for Mamba-Spike models.
"""

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset_loader import (
    prepare_nmnist_dataset,
    prepare_dvsgesture_dataset,
    prepare_cifar10dvs_dataset
)
from models.mamba_spike import (
    create_mamba_spike_nmnist,
    create_mamba_spike_dvsgesture,
    create_mamba_spike_cifar10dvs
)


class Evaluator:
    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = checkpoint_path
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.args = self.checkpoint['args']
        
        # Create output directory
        self.output_dir = os.path.dirname(checkpoint_path)
        
        # Load dataset
        self.test_loader, self.num_classes = self._load_test_dataset()
        
        # Create and load model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from epoch {self.checkpoint['epoch']} "
              f"with accuracy {self.checkpoint['accuracy']:.2f}%")
    
    def _load_test_dataset(self):
        """Load test dataset."""
        if self.args.dataset == 'nmnist':
            _, test_loader, num_classes = prepare_nmnist_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        elif self.args.dataset == 'dvsgesture':
            _, test_loader, num_classes = prepare_dvsgesture_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        elif self.args.dataset == 'cifar10dvs':
            _, test_loader, num_classes = prepare_cifar10dvs_dataset(
                data_dir=self.args.data_dir,
                batch_size=self.args.batch_size,
                time_window=self.args.time_window,
                dt=self.args.dt,
                num_workers=self.args.num_workers
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        return test_loader, num_classes
    
    def _create_model(self):
        """Create model based on dataset."""
        if self.args.dataset == 'nmnist':
            return create_mamba_spike_nmnist(num_classes=self.num_classes)
        elif self.args.dataset == 'dvsgesture':
            return create_mamba_spike_dvsgesture(num_classes=self.num_classes)
        elif self.args.dataset == 'cifar10dvs':
            return create_mamba_spike_cifar10dvs(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
    
    def evaluate(self):
        """Perform comprehensive evaluation."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Evaluating"):
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                
                # Get predictions
                pred = output.argmax(dim=1)
                
                # Store results
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean() * 100
        
        # Per-class accuracy
        cm = confusion_matrix(all_labels, all_preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        
        # Print results
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print("\nPer-class Accuracy:")
        for i in range(self.num_classes):
            print(f"  Class {i}: {per_class_acc[i]:.2f}%")
        
        # Save detailed results
        results = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                all_labels, all_preds, output_dict=True
            )
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        return accuracy, per_class_acc
    
    def _plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        
        plt.title(f'Confusion Matrix - {self.args.dataset.upper()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def analyze_temporal_dynamics(self, num_samples=10):
        """Analyze temporal dynamics of the model."""
        self.model.eval()
        
        # Get a few samples
        data_iter = iter(self.test_loader)
        data, labels = next(data_iter)
        data = data[:num_samples].to(self.device)
        labels = labels[:num_samples]
        
        # Hook to capture intermediate activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        
        # Register hook on spike-to-activation layer
        hook = self.model.spike_to_activation.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(data)
            predictions = output.argmax(dim=1).cpu()
        
        # Remove hook
        hook.remove()
        
        # Plot temporal dynamics
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            ax = axes[i]
            
            # Get activation for this sample
            act = activations[0][i]  # Shape: [time, features]
            
            # Plot mean activation over features
            mean_act = act.mean(dim=1).numpy()
            ax.plot(mean_act)
            ax.set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Mean Activation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_dynamics.png'), dpi=300)
        plt.close()
        
        print("Temporal dynamics analysis saved.")


def compare_results():
    """Compare results with paper's reported performance."""
    paper_results = {
        'nmnist': {
            'mamba_spike': 99.5,
            'baseline_snn': 98.8,
            'baseline_ann': 99.2
        },
        'dvsgesture': {
            'mamba_spike': 97.8,
            'baseline_snn': 96.5,
            'baseline_ann': 97.1
        },
        'cifar10dvs': {
            'mamba_spike': 78.1,
            'baseline_snn': 74.5,
            'baseline_ann': 76.8
        }
    }
    
    print("\nComparison with Paper Results:")
    print("-" * 50)
    print(f"{'Dataset':<15} {'Paper':<10} {'Ours':<10} {'Diff':<10}")
    print("-" * 50)
    
    # This would be filled in with actual results
    # For now, just showing the structure
    for dataset, results in paper_results.items():
        paper_acc = results['mamba_spike']
        print(f"{dataset:<15} {paper_acc:<10.1f} {'--':<10} {'--':<10}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mamba-Spike model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--analyze-temporal', action='store_true',
                        help='Analyze temporal dynamics')
    parser.add_argument('--compare-paper', action='store_true',
                        help='Compare with paper results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(args.checkpoint)
    
    # Run evaluation
    accuracy, per_class_acc = evaluator.evaluate()
    
    # Additional analysis
    if args.analyze_temporal:
        evaluator.analyze_temporal_dynamics()
    
    if args.compare_paper:
        compare_results()


if __name__ == "__main__":
    main()