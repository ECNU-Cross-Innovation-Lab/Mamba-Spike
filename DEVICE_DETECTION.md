# Automatic Device Detection

The training script now automatically detects and uses the best available computing device.

## Device Priority

The script automatically selects devices in this order:

1. **CUDA (NVIDIA GPUs)** - Highest priority
   - Used on: Windows, Linux with NVIDIA GPUs
   - Example: RTX 4090, RTX 3090, A100, etc.

2. **MPS (Apple Silicon GPU)** - Second priority
   - Used on: Mac with M1/M2/M3 chips
   - Provides GPU acceleration on Apple Silicon

3. **CPU** - Fallback
   - Used when no GPU is available
   - Slower but works everywhere

## How It Works

When you run training, the script will:

1. Check if CUDA is available and working
2. If CUDA fails, check if MPS is available
3. If neither GPU works, fall back to CPU
4. Display detailed device information

### Example Output

**On Windows/Linux with NVIDIA GPU:**
```
GPU detected: NVIDIA GeForce RTX 4090
GPU memory: 24.0 GB
CUDA version: 12.1
Using device: cuda
```

**On Mac with Apple Silicon:**
```
Apple Silicon GPU (MPS) detected
Using device: mps
```

**On CPU-only systems:**
```
No GPU detected, using CPU
Note: Training will be slower on CPU. Consider using a GPU for better performance.
Using device: cpu
```

## Testing Device Detection

Run the test script to verify your setup:

```bash
python test_device_detection.py
```

This will:
- Detect your best available device
- Show detailed hardware information
- Test basic tensor operations
- Confirm everything is working

## Usage

No changes needed! The training script automatically detects the best device:

```bash
# Works on all platforms - automatically uses best device
python train.py --dataset nmnist --epochs 50

# On Windows with RTX 4090
python train.py --dataset dvsgesture --batch-size 32 --epochs 100

# On Mac with M1/M2/M3
python train.py --dataset sequential_mnist --batch-size 16 --epochs 50

# On CPU (no GPU)
python train.py --dataset nmnist --batch-size 8 --epochs 50
```

## Error Handling

The script includes robust error handling:

### CUDA Initialization Error
If CUDA is available but fails to initialize (like the "shared object initialization failed" error):

```
Warning: CUDA available but initialization failed: [error details]
Falling back to CPU...
```

The script will automatically try MPS, then CPU.

### MPS Initialization Error
If MPS is available but fails:

```
Warning: MPS available but initialization failed: [error details]
Falling back to CPU...
```

The script will fall back to CPU.

### CPU Fallback
When using CPU:

```
No GPU detected, using CPU
Note: Training will be slower on CPU. Consider using a GPU for better performance.
```

## Recommended Batch Sizes by Device

### NVIDIA RTX 4090 (24GB VRAM)
```bash
python train.py --dataset nmnist --batch-size 64
python train.py --dataset dvsgesture --batch-size 32
python train.py --dataset cifar10dvs --batch-size 32
```

### NVIDIA RTX 3090 (24GB VRAM)
```bash
python train.py --dataset nmnist --batch-size 64
python train.py --dataset dvsgesture --batch-size 32
python train.py --dataset cifar10dvs --batch-size 32
```

### Apple M1/M2/M3 (8-16GB Unified Memory)
```bash
python train.py --dataset nmnist --batch-size 16
python train.py --dataset dvsgesture --batch-size 8
python train.py --dataset cifar10dvs --batch-size 8
```

### CPU (Any RAM)
```bash
python train.py --dataset nmnist --batch-size 4
python train.py --dataset sequential_mnist --batch-size 8
```

## Troubleshooting

### Issue: CUDA detected but training fails
**Solution:** Check CUDA version compatibility
```bash
python check_cuda.py
```
For RTX 4090, you need CUDA 11.8+

### Issue: MPS detected but training is slow
**Solution:** MPS is still maturing. Some operations fall back to CPU.
- Use smaller batch sizes
- Update to latest macOS and PyTorch

### Issue: CPU is too slow
**Solution:** Use a GPU-enabled system
- Cloud GPUs: Google Colab, AWS, Azure
- Local GPU: Install NVIDIA GPU or use Mac with Apple Silicon

## Platform-Specific Notes

### Windows
- Requires NVIDIA GPU + CUDA-enabled PyTorch
- See `WINDOWS_RTX4090_SETUP.md` for RTX 4090 setup

### macOS
- M1/M2/M3 automatically use MPS
- Intel Macs fall back to CPU

### Linux
- NVIDIA GPU + CUDA-enabled PyTorch for GPU acceleration
- Best for server/cluster training

## Performance Comparison

Approximate training speeds (N-MNIST, 1 epoch):

| Device | Time | Speedup |
|--------|------|---------|
| RTX 4090 | ~2 min | 20x |
| RTX 3090 | ~3 min | 13x |
| Apple M2 | ~10 min | 4x |
| CPU (i7) | ~40 min | 1x |

Times vary based on batch size and dataset.
