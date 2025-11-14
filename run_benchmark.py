#!/usr/bin/env python3
"""
Automated benchmark script for DVS Gesture and CIFAR10-DVS datasets.
Runs training for 4 epochs on each dataset and saves results.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
import argparse


class BenchmarkRunner:
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.results = {
            'timestamp': self.timestamp,
            'datasets': {}
        }

    def run_experiment(self, dataset, epochs=4, batch_size=None):
        """Run training experiment on a dataset."""
        print("\n" + "="*70)
        print(f"Running benchmark: {dataset.upper()}")
        print("="*70)

        # Prepare command
        cmd = [
            sys.executable, "train.py",
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--output-dir", os.path.join(self.run_dir, dataset)
        ]

        # Add batch size if specified
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])

        print(f"\nCommand: {' '.join(cmd)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Record start time
        start_time = time.time()

        # Run training
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Record end time
            end_time = time.time()
            duration = end_time - start_time

            # Parse results
            success = True
            stdout = result.stdout
            stderr = result.stderr

            print(f"\n✅ Training completed successfully!")
            print(f"Duration: {duration/60:.2f} minutes")

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time
            success = False
            stdout = e.stdout
            stderr = e.stderr

            print(f"\n❌ Training failed!")
            print(f"Error: {stderr[-500:]}")  # Last 500 chars of error

        # Extract metrics from stdout
        metrics = self._parse_training_output(stdout)

        # Save results
        result_data = {
            'dataset': dataset,
            'epochs': epochs,
            'batch_size': batch_size,
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'success': success,
            'metrics': metrics,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat()
        }

        self.results['datasets'][dataset] = result_data

        # Save individual log
        log_file = os.path.join(self.run_dir, f"{dataset}_log.txt")
        with open(log_file, 'w') as f:
            f.write("=== STDOUT ===\n")
            f.write(stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(stderr)

        print(f"Log saved to: {log_file}")

        return result_data

    def _parse_training_output(self, stdout):
        """Parse training metrics from stdout."""
        metrics = {
            'final_train_acc': None,
            'final_test_acc': None,
            'best_test_acc': None,
            'epoch_results': []
        }

        lines = stdout.split('\n')

        for line in lines:
            # Parse epoch results
            if 'Train Loss:' in line and 'Train Acc:' in line:
                try:
                    # Extract train accuracy
                    train_acc_str = line.split('Train Acc:')[1].split('%')[0].strip()
                    train_acc = float(train_acc_str)

                    # Extract test accuracy if present
                    if 'Test Acc:' in line:
                        test_acc_str = line.split('Test Acc:')[1].split('%')[0].strip()
                        test_acc = float(test_acc_str)
                    else:
                        test_acc = None

                    metrics['epoch_results'].append({
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    })

                    # Update final accuracies
                    metrics['final_train_acc'] = train_acc
                    if test_acc is not None:
                        metrics['final_test_acc'] = test_acc
                except:
                    pass

            # Parse best accuracy
            if 'Best test accuracy:' in line:
                try:
                    best_acc_str = line.split('Best test accuracy:')[1].split('%')[0].strip()
                    metrics['best_test_acc'] = float(best_acc_str)
                except:
                    pass

        return metrics

    def save_summary(self):
        """Save benchmark summary."""
        summary_file = os.path.join(self.run_dir, "benchmark_summary.json")

        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📊 Summary saved to: {summary_file}")

        return summary_file

    def print_summary(self):
        """Print benchmark summary to console."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)

        for dataset, data in self.results['datasets'].items():
            print(f"\n{dataset.upper()}")
            print("-" * 50)
            print(f"Status: {'✅ Success' if data['success'] else '❌ Failed'}")
            print(f"Duration: {data['duration_minutes']:.2f} minutes")

            if data['success'] and data['metrics']['final_train_acc'] is not None:
                print(f"Final Train Accuracy: {data['metrics']['final_train_acc']:.2f}%")
                if data['metrics']['final_test_acc'] is not None:
                    print(f"Final Test Accuracy: {data['metrics']['final_test_acc']:.2f}%")
                if data['metrics']['best_test_acc'] is not None:
                    print(f"Best Test Accuracy: {data['metrics']['best_test_acc']:.2f}%")

                # Show epoch progression
                print("\nEpoch progression:")
                for i, epoch_data in enumerate(data['metrics']['epoch_results'], 1):
                    test_str = f", Test: {epoch_data['test_acc']:.2f}%" if epoch_data['test_acc'] else ""
                    print(f"  Epoch {i}: Train: {epoch_data['train_acc']:.2f}%{test_str}")

        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Run benchmark on DVS Gesture and CIFAR10-DVS')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs per dataset')
    parser.add_argument('--batch-size-dvsgesture', type=int, default=2,
                       help='Batch size for DVS Gesture')
    parser.add_argument('--batch-size-cifar10dvs', type=int, default=2,
                       help='Batch size for CIFAR10-DVS')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    args = parser.parse_args()

    print("="*70)
    print("MAMBA-SPIKE BENCHMARK")
    print("="*70)
    print(f"Datasets: DVS Gesture, CIFAR10-DVS")
    print(f"Epochs per dataset: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)

    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=args.output_dir)

    # Run benchmarks
    datasets_config = [
        ('dvsgesture', args.batch_size_dvsgesture),
        ('cifar10dvs', args.batch_size_cifar10dvs)
    ]

    for dataset, batch_size in datasets_config:
        runner.run_experiment(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=batch_size
        )

    # Save and print summary
    runner.save_summary()
    runner.print_summary()

    print(f"\n✅ Benchmark completed!")
    print(f"📁 Results saved in: {runner.run_dir}")


if __name__ == "__main__":
    main()
