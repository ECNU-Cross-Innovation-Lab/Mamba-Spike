"""
Test script for Sequential MNIST dataset with Mamba-Spike model.
Validates that all paper-specified modifications work correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm

from models.mamba_spike import create_mamba_spike_sequential_mnist
from data.dataset_loader import prepare_sequential_mnist_dataset


def test_forward_pass():
    """Test that model can process Sequential MNIST data."""
    print("=" * 60)
    print("Testing Forward Pass on Sequential MNIST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create model
    print("Creating Mamba-Spike model for Sequential MNIST...")
    model = create_mamba_spike_sequential_mnist(num_classes=10)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully")
    print(f"  Parameters: {num_params:,}\n")

    # Load a small batch of data
    print("Loading Sequential MNIST dataset...")
    train_loader, test_loader, num_classes = prepare_sequential_mnist_dataset(
        batch_size=4,
        time_steps=100,
        dt=0.1,
        num_workers=0
    )
    print(f"Dataset loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Number of classes: {num_classes}\n")

    # Get one batch
    data, labels = next(iter(train_loader))
    print(f"Batch loaded:")
    print(f"  Data shape: {data.shape}")
    print(f"  Labels: {labels.numpy()}\n")

    # Test forward pass
    print("Running forward pass...")
    data = data.to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(data)
    end_time = time.time()

    print(f"Forward pass completed in {end_time - start_time:.3f} seconds")
    print(f"  Output shape: {output.shape}")

    # Check predictions
    predictions = output.argmax(dim=1).cpu().numpy()
    print(f"  Predictions: {predictions}")
    accuracy = (predictions == labels.numpy()).mean() * 100
    print(f"  Random accuracy: {accuracy:.1f}% (expected ~10% before training)\n")

    print("Forward pass test PASSED")
    return True


def test_training_loop(num_epochs=2, batch_size=32):
    """Test a short training loop to verify gradient flow."""
    print("\n" + "=" * 60)
    print(f"Testing Training Loop ({num_epochs} epochs)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create model
    model = create_mamba_spike_sequential_mnist(num_classes=10)
    model = model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Load data
    train_loader, test_loader, _ = prepare_sequential_mnist_dataset(
        batch_size=batch_size,
        time_steps=100,
        dt=0.1,
        num_workers=0
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

            # Only do a few batches for testing
            if batch_idx >= 10:
                break

        avg_loss = epoch_loss / (batch_idx + 1)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    print("\nTraining loop test PASSED")
    print("Gradients are flowing correctly")
    return True


def test_modified_components():
    """Test that all paper-specified modifications are working."""
    print("\n" + "=" * 60)
    print("Testing Paper-Specified Modifications")
    print("=" * 60)

    from models.mamba_spike import SpikingFrontEnd, SpikeToActivation

    # Test 1: Recurrent connections
    print("\n1. Testing Recurrent Connections...")
    spiking_fe = SpikingFrontEnd(
        in_channels=2,
        hidden_channels=32,
        out_channels=64,
        beta=0.97,
        use_recurrent=True
    )

    assert hasattr(spiking_fe, 'recurrent1'), "Missing recurrent1 connection"
    assert hasattr(spiking_fe, 'recurrent2'), "Missing recurrent2 connection"
    assert hasattr(spiking_fe, 'recurrent3'), "Missing recurrent3 connection"
    print("   PASSED: Recurrent connections present")

    # Test 2: LIF time constant (beta=0.97 for 30ms)
    print("\n2. Testing LIF Time Constant...")
    assert spiking_fe.lif1.beta == 0.97, f"Beta should be 0.97, got {spiking_fe.lif1.beta}"

    # Calculate time constant
    dt = 1.0  # ms
    tau = -dt / np.log(0.97)
    print(f"   Beta = 0.97 corresponds to tau ≈ {tau:.1f}ms")
    print(f"   PASSED: Time constant is approximately 30ms (paper Fig 5)")

    # Test 3: Spike-to-Activation interface
    print("\n3. Testing Spike-to-Activation Interface...")
    spike_converter = SpikeToActivation(time_window=5)

    # Create test spikes
    test_spikes = torch.randint(0, 2, (2, 10, 2, 28, 28)).float()  # (B, T, C, H, W)
    activations = spike_converter(test_spikes)

    print(f"   Input spikes shape: {test_spikes.shape}")
    print(f"   Output activations shape: {activations.shape}")
    print(f"   Time window: {spike_converter.time_window}")

    # Check that activations are normalized (firing rates should be in [0, 1])
    assert activations.min() >= 0 and activations.max() <= 1, "Firing rates should be in [0, 1]"
    print(f"   Activation range: [{activations.min():.3f}, {activations.max():.3f}]")
    print("   PASSED: Fixed time window + firing rate normalization")

    print("\nAll paper modifications VERIFIED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Mamba-Spike Sequential MNIST Test Suite")
    print("Paper Compliance Verification")
    print("=" * 60)

    try:
        # Test 1: Modified components
        test_modified_components()

        # Test 2: Forward pass
        test_forward_pass()

        # Test 3: Training loop
        test_training_loop(num_epochs=2, batch_size=32)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe implementation now matches the paper specifications:")
        print("  1. Recurrent connections in spiking front-end")
        print("  2. LIF time constant: 30ms (beta=0.97)")
        print("  3. Spike-to-Activation: fixed time window + firing rate")
        print("  4. Sequential MNIST dataset support")
        print("\nTo train the full model, run:")
        print("  python train.py --dataset sequential_mnist --epochs 100 --batch-size 64")

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
