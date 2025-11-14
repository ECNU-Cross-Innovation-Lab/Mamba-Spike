# Windows RTX 4090 Setup Guide

## Error: "CUDA error: shared object initialization failed"

This error on RTX 4090 is usually caused by CUDA version incompatibility. RTX 4090 (Ada Lovelace architecture) requires CUDA 11.8 or higher.

---

## Quick Diagnosis

First, run the diagnostics script:

```bash
python check_cuda.py
```

This will tell you exactly what's wrong.

---

## Solution 1: Upgrade PyTorch with CUDA 12.1 (Recommended)

RTX 4090 works best with CUDA 12.1. Follow these steps:

### Step 1: Uninstall old PyTorch
```bash
conda activate pcds_gpu
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install PyTorch with CUDA 12.1

**Option A: Using conda (Recommended)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Option B: Using pip**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Should output:
```
CUDA available: True
CUDA version: 12.1
```

---

## Solution 2: Update NVIDIA Drivers

RTX 4090 requires driver version 525.60.11 or higher.

### Check current driver version:
```bash
nvidia-smi
```

Look for "Driver Version" in the output.

### Update drivers:
1. Download latest drivers from: https://www.nvidia.com/Download/index.aspx
2. Select:
   - Product Type: GeForce
   - Product Series: GeForce RTX 40 Series
   - Product: GeForce RTX 4090
   - Operating System: Windows 11 (or your version)
3. Install and restart

---

## Solution 3: Reduce Memory Usage

If CUDA initializes but crashes during training, try reducing batch size:

```bash
python train.py --dataset dvsgesture --epochs 1 --batch-size 8
```

Or even smaller:
```bash
python train.py --dataset dvsgesture --epochs 1 --batch-size 4
```

---

## Solution 4: Environment Variables (Advanced)

If problems persist, set these environment variables before running:

**PowerShell:**
```powershell
$env:CUDA_LAUNCH_BLOCKING=1
$env:TORCH_USE_CUDA_DSA=1
python train.py --dataset dvsgesture --epochs 1
```

**CMD:**
```cmd
set CUDA_LAUNCH_BLOCKING=1
set TORCH_USE_CUDA_DSA=1
python train.py --dataset dvsgesture --epochs 1
```

This enables synchronous CUDA operations and device-side assertions for better error messages.

---

## Verify Your Setup

After applying fixes, test with:

```bash
# 1. Run diagnostics
python check_cuda.py

# 2. Quick test with small batch
python train.py --dataset sequential_mnist --epochs 1 --batch-size 4

# 3. Test DVS Gesture with reduced batch
python train.py --dataset dvsgesture --epochs 1 --batch-size 8
```

---

## Recommended Configuration for RTX 4090

```bash
# Install commands
conda create -n mamba_spike python=3.9
conda activate mamba_spike
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Training commands (optimized for 4090's 24GB VRAM)
python train.py --dataset nmnist --batch-size 64 --epochs 50
python train.py --dataset dvsgesture --batch-size 32 --epochs 100
python train.py --dataset cifar10dvs --batch-size 32 --epochs 100
```

---

## Troubleshooting Checklist

- [ ] CUDA version >= 11.8 (check with `python check_cuda.py`)
- [ ] NVIDIA driver >= 525.60.11 (check with `nvidia-smi`)
- [ ] PyTorch installed with CUDA support (not CPU-only)
- [ ] No other programs heavily using GPU (close games, other ML training)
- [ ] Windows up to date with latest updates
- [ ] Sufficient disk space for dataset caching

---

## Common Issues

### Issue 1: "CUDA out of memory"
**Solution:** Reduce batch size: `--batch-size 4` or `--batch-size 8`

### Issue 2: "No CUDA-capable device is detected"
**Solution:**
1. Update NVIDIA drivers
2. Check Device Manager that GPU is recognized
3. Reinstall CUDA toolkit

### Issue 3: Training is very slow
**Solution:**
1. Check GPU utilization: `nvidia-smi -l 1`
2. If GPU usage is low, increase batch size
3. Ensure using GPU, not CPU: check "Using device: cuda" in output

### Issue 4: "cuDNN error"
**Solution:** Reinstall PyTorch with matching cuDNN:
```bash
pip uninstall torch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## Need More Help?

If issues persist after trying all solutions:

1. Run `python check_cuda.py` and save the output
2. Run `nvidia-smi` and save the output
3. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
4. Open an issue with all this information

---

## References

- PyTorch CUDA compatibility: https://pytorch.org/get-started/locally/
- NVIDIA RTX 4090 specs: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- CUDA downloads: https://developer.nvidia.com/cuda-downloads
