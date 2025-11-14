#!/usr/bin/env python3
"""
CUDA Diagnostics Script for Windows RTX 4090
Checks PyTorch, CUDA, and GPU compatibility
"""

import sys
import torch

print("="*60)
print("CUDA Diagnostics for RTX 4090")
print("="*60)

# 1. PyTorch and CUDA versions
print("\n1. PyTorch Configuration:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version (PyTorch): {torch.version.cuda}")
    print(f"   cuDNN version: {torch.backends.cudnn.version()}")
    print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
else:
    print("   ⚠️  CUDA is NOT available!")
    print("\n   Possible reasons:")
    print("   - PyTorch CPU-only version installed")
    print("   - NVIDIA drivers not installed")
    print("   - CUDA toolkit not installed")
    sys.exit(1)

# 2. GPU Information
print("\n2. GPU Information:")
try:
    print(f"   GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Compute Capability: {torch.cuda.get_device_capability(i)}")

        # Memory info
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   - Total Memory: {total_mem:.2f} GB")

        # Current allocation
        if torch.cuda.memory_allocated(i) > 0:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            print(f"   - Allocated Memory: {allocated:.2f} GB")
except Exception as e:
    print(f"   ❌ Error getting GPU info: {e}")

# 3. RTX 4090 Specific Requirements
print("\n3. RTX 4090 Requirements:")
print("   Minimum CUDA version: 11.8")
print("   Minimum driver version: 525.60.11 (Windows)")
print("   Compute Capability: 8.9")

cuda_version = torch.version.cuda
if cuda_version:
    major, minor = map(int, cuda_version.split('.')[:2])
    cuda_ver_num = major + minor / 10

    if cuda_ver_num >= 11.8:
        print(f"   ✅ CUDA {cuda_version} is compatible with RTX 4090")
    else:
        print(f"   ❌ CUDA {cuda_version} is TOO OLD for RTX 4090!")
        print(f"      You need CUDA 11.8 or higher")

# 4. Simple CUDA test
print("\n4. CUDA Functionality Test:")
try:
    # Test basic tensor operations
    device = torch.device('cuda:0')
    print(f"   Testing tensor creation on {device}...")

    x = torch.randn(10, 10, device=device)
    print("   ✅ Tensor creation successful")

    print("   Testing tensor operation...")
    y = x @ x.t()
    print("   ✅ Matrix multiplication successful")

    print("   Testing conv2d operation...")
    conv = torch.nn.Conv2d(3, 64, 3).cuda()
    test_input = torch.randn(1, 3, 32, 32).cuda()
    output = conv(test_input)
    print("   ✅ Conv2d operation successful")

    print("\n✅ All CUDA tests passed!")

except RuntimeError as e:
    print(f"\n❌ CUDA test failed: {e}")
    print("\nThis error suggests:")
    print("1. Driver/CUDA version mismatch")
    print("2. Corrupted CUDA installation")
    print("3. GPU hardware issue")

# 5. Recommendations
print("\n" + "="*60)
print("Recommendations:")
print("="*60)

if cuda_version:
    major, minor = map(int, cuda_version.split('.')[:2])
    cuda_ver_num = major + minor / 10

    if cuda_ver_num < 11.8:
        print("\n⚠️  ACTION REQUIRED: Upgrade PyTorch with CUDA 11.8+")
        print("\n   For Windows + RTX 4090, install PyTorch with CUDA 12.1:")
        print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("\n   Or with pip:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("\n✅ CUDA version looks good")
        print("\nIf you're still getting errors, try:")
        print("1. Update NVIDIA drivers to latest version")
        print("2. Reinstall PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("3. Reduce batch size in train.py (--batch-size 8 or 4)")
        print("4. Check GPU with: nvidia-smi")

print("\n" + "="*60)
