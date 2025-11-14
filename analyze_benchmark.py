#!/usr/bin/env python3
"""
Analyze and visualize benchmark results.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(result_dir):
    """Load benchmark results from JSON file."""
    summary_file = os.path.join(result_dir, "benchmark_summary.json")

    if not os.path.exists(summary_file):
        print(f"Error: {summary_file} not found!")
        return None

    with open(summary_file, 'r') as f:
        results = json.load(f)

    return results


def plot_training_curves(results, output_dir):
    """Plot training curves for all datasets."""
    datasets = results['datasets']
    num_datasets = len(datasets)

    if num_datasets == 0:
        print("No datasets to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, num_datasets, figsize=(8*num_datasets, 6))

    if num_datasets == 1:
        axes = [axes]

    for idx, (dataset_name, dataset_data) in enumerate(datasets.items()):
        ax = axes[idx]

        if not dataset_data['success']:
            ax.text(0.5, 0.5, 'Training Failed',
                   ha='center', va='center', fontsize=16)
            ax.set_title(f"{dataset_name.upper()}")
            continue

        # Extract epoch data
        epoch_results = dataset_data['metrics']['epoch_results']
        if not epoch_results:
            ax.text(0.5, 0.5, 'No data available',
                   ha='center', va='center', fontsize=16)
            ax.set_title(f"{dataset_name.upper()}")
            continue

        epochs = list(range(1, len(epoch_results) + 1))
        train_accs = [ep['train_acc'] for ep in epoch_results]
        test_accs = [ep['test_acc'] for ep in epoch_results if ep['test_acc'] is not None]

        # Plot training accuracy
        ax.plot(epochs, train_accs, 'o-', linewidth=2, markersize=8,
               label='Train Accuracy', color='#2E86AB')

        # Plot test accuracy if available
        if test_accs and len(test_accs) == len(epochs):
            ax.plot(epochs, test_accs, 's-', linewidth=2, markersize=8,
                   label='Test Accuracy', color='#A23B72')

        # Styling
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f"{dataset_name.upper()}", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

        # Set y-axis range
        ax.set_ylim(0, 100)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {output_file}")

    return output_file


def plot_comparison(results, output_dir):
    """Plot comparison bar chart of final accuracies."""
    datasets = results['datasets']

    dataset_names = []
    train_accs = []
    test_accs = []
    durations = []

    for dataset_name, dataset_data in datasets.items():
        if dataset_data['success']:
            dataset_names.append(dataset_name.upper())
            train_accs.append(dataset_data['metrics']['final_train_acc'] or 0)
            test_accs.append(dataset_data['metrics']['best_test_acc'] or
                           dataset_data['metrics']['final_test_acc'] or 0)
            durations.append(dataset_data['duration_minutes'])

    if not dataset_names:
        print("No successful runs to compare")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy comparison
    x = np.arange(len(dataset_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, train_accs, width, label='Train Accuracy',
                    color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_accs, width, label='Test Accuracy',
                    color='#A23B72', alpha=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10)

    # Plot 2: Training duration
    bars3 = ax2.bar(dataset_names, durations, color='#F18F01', alpha=0.8)

    ax2.set_ylabel('Duration (minutes)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Duration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} min',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, "comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison chart saved to: {output_file}")

    return output_file


def generate_report(results, output_dir):
    """Generate a markdown report of the benchmark results."""
    report_file = os.path.join(output_dir, "BENCHMARK_REPORT.md")

    with open(report_file, 'w') as f:
        f.write("# Benchmark Results Report\n\n")
        f.write(f"**Run Date:** {results['timestamp']}\n\n")

        f.write("## Summary\n\n")

        # Create summary table
        f.write("| Dataset | Status | Duration | Final Train Acc | Best Test Acc |\n")
        f.write("|---------|--------|----------|-----------------|---------------|\n")

        for dataset_name, dataset_data in results['datasets'].items():
            status = "✅ Success" if dataset_data['success'] else "❌ Failed"
            duration = f"{dataset_data['duration_minutes']:.2f} min"

            if dataset_data['success'] and dataset_data['metrics']['final_train_acc']:
                train_acc = f"{dataset_data['metrics']['final_train_acc']:.2f}%"
                test_acc = f"{dataset_data['metrics']['best_test_acc']:.2f}%" if dataset_data['metrics']['best_test_acc'] else "N/A"
            else:
                train_acc = "N/A"
                test_acc = "N/A"

            f.write(f"| {dataset_name.upper()} | {status} | {duration} | {train_acc} | {test_acc} |\n")

        f.write("\n## Detailed Results\n\n")

        # Detailed results for each dataset
        for dataset_name, dataset_data in results['datasets'].items():
            f.write(f"### {dataset_name.upper()}\n\n")

            if not dataset_data['success']:
                f.write("❌ Training failed\n\n")
                continue

            f.write(f"- **Duration:** {dataset_data['duration_minutes']:.2f} minutes\n")
            f.write(f"- **Batch Size:** {dataset_data['batch_size']}\n")
            f.write(f"- **Epochs:** {dataset_data['epochs']}\n\n")

            metrics = dataset_data['metrics']
            if metrics['final_train_acc']:
                f.write(f"**Final Training Accuracy:** {metrics['final_train_acc']:.2f}%\n\n")
            if metrics['final_test_acc']:
                f.write(f"**Final Test Accuracy:** {metrics['final_test_acc']:.2f}%\n\n")
            if metrics['best_test_acc']:
                f.write(f"**Best Test Accuracy:** {metrics['best_test_acc']:.2f}%\n\n")

            # Epoch-by-epoch results
            if metrics['epoch_results']:
                f.write("**Epoch-by-Epoch Results:**\n\n")
                f.write("| Epoch | Train Acc | Test Acc |\n")
                f.write("|-------|-----------|----------|\n")

                for i, epoch_data in enumerate(metrics['epoch_results'], 1):
                    train_acc = f"{epoch_data['train_acc']:.2f}%"
                    test_acc = f"{epoch_data['test_acc']:.2f}%" if epoch_data['test_acc'] else "N/A"
                    f.write(f"| {i} | {train_acc} | {test_acc} |\n")

                f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        f.write("![Comparison](comparison.png)\n\n")

        f.write("---\n")
        f.write("*Generated by Mamba-Spike benchmark script*\n")

    print(f"📄 Report saved to: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('result_dir', type=str,
                       help='Directory containing benchmark_summary.json')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.result_dir}")
    results = load_benchmark_results(args.result_dir)

    if results is None:
        return

    print(f"Found {len(results['datasets'])} datasets")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(results, args.result_dir)
    plot_comparison(results, args.result_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(results, args.result_dir)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
