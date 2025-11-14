#!/usr/bin/env python3
"""
Test script for automatic device detection.
This script tests the device detection logic without running full training.
"""

import torch


def test_device_detection():
    """Test automatic device detection with detailed output."""
    print("="*60)
    print("Device Detection Test")
    print("="*60)

    # Test CUDA
    print("\n1. Testing CUDA (NVIDIA GPU):")
    if torch.cuda.is_available():
        print("   ✅ CUDA is available")
        try:
            device = torch.device("cuda")
            test_tensor = torch.zeros(1).to(device)

            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_version = torch.version.cuda

            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   CUDA Version: {cuda_version}")
            print(f"   ✅ CUDA initialization successful")

            selected_device = "cuda"
        except Exception as e:
            print(f"   ❌ CUDA initialization failed: {e}")
            selected_device = None
    else:
        print("   ❌ CUDA not available")
        selected_device = None

    # Test MPS
    if selected_device is None:
        print("\n2. Testing MPS (Apple Silicon GPU):")
        if torch.backends.mps.is_available():
            print("   ✅ MPS is available")
            try:
                device = torch.device("mps")
                test_tensor = torch.zeros(1).to(device)
                print(f"   ✅ MPS initialization successful")
                selected_device = "mps"
            except Exception as e:
                print(f"   ❌ MPS initialization failed: {e}")
                selected_device = None
        else:
            print("   ❌ MPS not available")

    # Fallback to CPU
    if selected_device is None:
        print("\n3. Falling back to CPU:")
        device = torch.device("cpu")
        print("   ✅ CPU will be used")
        selected_device = "cpu"

    # Final result
    print("\n" + "="*60)
    print("Selected Device:", selected_device.upper())
    print("="*60)

    # Performance note
    if selected_device == "cpu":
        print("\n⚠️  Note: Training on CPU will be much slower than GPU.")
        print("   Consider using a machine with CUDA or MPS support for faster training.")
    elif selected_device == "cuda":
        print("\n✅ CUDA GPU detected - training will be fast!")
    elif selected_device == "mps":
        print("\n✅ Apple Silicon GPU detected - training will be accelerated!")

    return selected_device


def test_simple_operation():
    """Test a simple tensor operation on the detected device."""
    print("\n" + "="*60)
    print("Testing Simple Operations")
    print("="*60)

    device = test_device_detection()
    device_obj = torch.device(device)

    try:
        print(f"\nTesting matrix multiplication on {device}...")
        x = torch.randn(100, 100).to(device_obj)
        y = torch.randn(100, 100).to(device_obj)
        z = torch.matmul(x, y)
        print(f"✅ Matrix multiplication successful")
        print(f"   Result shape: {z.shape}")

        print(f"\nTesting convolution on {device}...")
        conv = torch.nn.Conv2d(3, 64, 3).to(device_obj)
        input_tensor = torch.randn(1, 3, 32, 32).to(device_obj)
        output = conv(input_tensor)
        print(f"✅ Convolution successful")
        print(f"   Output shape: {output.shape}")

        print("\n" + "="*60)
        print("✅ All operations completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        print("\nThis might indicate a problem with your PyTorch installation.")
        print("Consider reinstalling PyTorch for your platform.")


if __name__ == "__main__":
    test_simple_operation()
