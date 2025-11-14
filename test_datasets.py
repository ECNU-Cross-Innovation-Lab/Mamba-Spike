#!/usr/bin/env python3
"""
Test script to verify all dataset loaders work correctly.
This script will attempt to load one batch from each dataset to ensure
the data pipeline is functioning properly.
"""

import sys
import torch
from data.dataset_loader import (
    prepare_nmnist_dataset,
    prepare_dvsgesture_dataset,
    prepare_cifar10dvs_dataset,
    prepare_sequential_mnist_dataset
)


def test_dataset(dataset_name, prepare_fn, **kwargs):
    """Test a single dataset by loading one batch."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} dataset...")
    print(f"{'='*60}")

    try:
        # Prepare dataset with minimal batch size and workers for testing
        train_loader, test_loader, num_classes = prepare_fn(
            batch_size=2,  # Small batch size for quick testing
            num_workers=0,  # No multiprocessing for testing
            **kwargs
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")

        # Try to get one batch from train loader
        print(f"\nLoading one training batch...")
        train_data, train_labels = next(iter(train_loader))
        print(f"✓ Training batch loaded")
        print(f"  - Data shape: {train_data.shape}")
        print(f"  - Labels shape: {train_labels.shape}")
        print(f"  - Data type: {train_data.dtype}")
        print(f"  - Data range: [{train_data.min():.3f}, {train_data.max():.3f}]")

        # Try to get one batch from test loader
        print(f"\nLoading one test batch...")
        test_data, test_labels = next(iter(test_loader))
        print(f"✓ Test batch loaded")
        print(f"  - Data shape: {test_data.shape}")
        print(f"  - Labels shape: {test_labels.shape}")

        print(f"\n✅ {dataset_name} passed all tests!")
        return True

    except Exception as e:
        print(f"\n❌ {dataset_name} failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all datasets."""
    print("="*60)
    print("Dataset Loader Test Suite")
    print("="*60)
    print("\nThis will test all neuromorphic dataset loaders.")
    print("Note: First run will download datasets (may take time).")

    results = {}

    # Test Sequential MNIST (smallest, fastest to test)
    results['Sequential MNIST'] = test_dataset(
        'Sequential MNIST',
        prepare_sequential_mnist_dataset,
        time_steps=20,  # Reduced for faster testing
        dt=0.1
    )

    # Test N-MNIST
    results['N-MNIST'] = test_dataset(
        'N-MNIST',
        prepare_nmnist_dataset,
        time_window=300000,
        dt=10000  # Fewer time bins for faster testing
    )

    # Test DVS Gesture
    results['DVS Gesture'] = test_dataset(
        'DVS Gesture',
        prepare_dvsgesture_dataset,
        time_window=500000,
        dt=50000  # Fewer time bins for faster testing
    )

    # Test CIFAR10-DVS
    results['CIFAR10-DVS'] = test_dataset(
        'CIFAR10-DVS',
        prepare_cifar10dvs_dataset,
        time_window=1000000,
        dt=100000  # Fewer time bins for faster testing
    )

    # Print summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for dataset, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{dataset:20s} {status}")

    print(f"\n{passed}/{total} datasets passed")

    if passed == total:
        print("\n🎉 All datasets working correctly!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} dataset(s) failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
