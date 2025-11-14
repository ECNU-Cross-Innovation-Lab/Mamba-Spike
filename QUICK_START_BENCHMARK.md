# Quick Start: Running Benchmarks

## 🚀 One Command to Test Everything

### Windows
```cmd
run_benchmark.bat
```

### Linux/Mac
```bash
./run_benchmark.sh
```

This will automatically:
1. ✅ Detect your device (CUDA/MPS/CPU)
2. ✅ Train on DVS Gesture (4 epochs)
3. ✅ Train on CIFAR10-DVS (4 epochs)
4. ✅ Save all results
5. ✅ Generate summary

**Estimated time:** 20-40 minutes (with GPU) or 2-4 hours (CPU only)

---

## 📊 View Results

After benchmark completes, you'll see output like:

```
BENCHMARK RESULTS SUMMARY
======================================================================

DVSGESTURE
--------------------------------------------------
Status: ✅ Success
Duration: 12.34 minutes
Final Train Accuracy: 75.23%
Final Test Accuracy: 72.45%
Best Test Accuracy: 73.12%

Epoch progression:
  Epoch 1: Train: 55.23%, Test: 52.45%
  Epoch 2: Train: 65.12%, Test: 62.34%
  Epoch 3: Train: 72.45%, Test: 70.12%
  Epoch 4: Train: 75.23%, Test: 72.45%

CIFAR10DVS
--------------------------------------------------
Status: ✅ Success
Duration: 15.67 minutes
Final Train Accuracy: 45.67%
Final Test Accuracy: 42.34%
Best Test Accuracy: 43.21%

Epoch progression:
  Epoch 1: Train: 25.34%, Test: 23.12%
  Epoch 2: Train: 35.67%, Test: 32.45%
  Epoch 3: Train: 42.12%, Test: 39.67%
  Epoch 4: Train: 45.67%, Test: 42.34%
```

---

## 📈 Generate Visual Reports

Find your results directory:

```bash
# List recent runs
ls -lt benchmark_results/

# You'll see something like:
# run_20250114_143022/
```

Generate visualizations:

```bash
python analyze_benchmark.py benchmark_results/run_20250114_143022/
```

This creates:
- 📊 `training_curves.png` - Training progress charts
- 📊 `comparison.png` - Side-by-side comparison
- 📄 `BENCHMARK_REPORT.md` - Complete analysis

---

## 🎯 What the Results Mean

### Accuracy Expectations (4 epochs only)

| Dataset | Expected Range | Notes |
|---------|---------------|-------|
| DVS Gesture | 70-80% | 4 epochs is minimal training |
| CIFAR10-DVS | 40-50% | Harder dataset, needs more epochs |

**Important:** 4 epochs is just for quick testing!
- For real results, use 50-100 epochs
- Paper results: DVS Gesture 96.8%, CIFAR10-DVS 78.9%

### Training Time

| Hardware | DVS Gesture | CIFAR10-DVS | Total |
|----------|-------------|-------------|-------|
| RTX 4090 | ~5 min | ~8 min | ~13 min |
| RTX 3090 | ~8 min | ~12 min | ~20 min |
| Apple M2 | ~20 min | ~30 min | ~50 min |
| CPU (i7) | ~2 hours | ~3 hours | ~5 hours |

---

## 🛠️ Customization

### Run More Epochs (Better Results)

```bash
python run_benchmark.py --epochs 20
```

### Adjust for Your Hardware

**RTX 4090 (24GB):**
```bash
python run_benchmark.py --epochs 20 \
    --batch-size-dvsgesture 32 \
    --batch-size-cifar10dvs 32
```

**Apple M2 (16GB):**
```bash
python run_benchmark.py --epochs 10 \
    --batch-size-dvsgesture 8 \
    --batch-size-cifar10dvs 8
```

**CPU Only:**
```bash
python run_benchmark.py --epochs 2 \
    --batch-size-dvsgesture 4 \
    --batch-size-cifar10dvs 4
```

---

## ✅ What to Check

### Before Running

1. **Check Device:**
   ```bash
   python test_device_detection.py
   ```
   Should show CUDA or MPS detected (or CPU as fallback)

2. **Check CUDA (Windows only):**
   ```bash
   python check_cuda.py
   ```
   If RTX 4090, need CUDA 11.8+

3. **Check Datasets:**
   ```bash
   ls data/
   ```
   Should see `dvsgesture/` and `cifar10dvs/` (or will download automatically)

### During Training

Watch for:
- ✅ "Using device: cuda" or "Using device: mps" (good!)
- ❌ "Using device: cpu" (slow, but works)
- Progress bars showing epoch completion
- Accuracy increasing over epochs

### After Training

Check results:
```bash
# See summary
cat benchmark_results/run_TIMESTAMP/benchmark_summary.json

# View logs
cat benchmark_results/run_TIMESTAMP/dvsgesture_log.txt
cat benchmark_results/run_TIMESTAMP/cifar10dvs_log.txt
```

---

## 🐛 Troubleshooting

### Error: CUDA initialization failed
**Fix:** See `WINDOWS_RTX4090_SETUP.md`

### Error: Out of memory
**Fix:** Reduce batch size
```bash
python run_benchmark.py --batch-size-dvsgesture 8 --batch-size-cifar10dvs 8
```

### Error: Dataset not found
**Fix:** Script will auto-download, but needs internet connection

### Training is very slow
**Check:**
1. Are you using GPU? Run `python test_device_detection.py`
2. Is GPU being utilized? Run `nvidia-smi -l 1` (NVIDIA only)
3. Try smaller batch size if memory is maxed out

---

## 📚 Learn More

- **Complete guide:** `BENCHMARK_GUIDE.md`
- **Device detection:** `DEVICE_DETECTION.md`
- **RTX 4090 setup:** `WINDOWS_RTX4090_SETUP.md`
- **Main README:** `README.md`

---

## 💡 Quick Tips

1. **First time?** Run with 4 epochs to test everything works
2. **Real training?** Use 50-100 epochs for paper-quality results
3. **Save results!** Each run creates a timestamped directory
4. **Compare runs:** Keep multiple `benchmark_results/run_*/` folders
5. **Use tensorboard:** View detailed training curves
   ```bash
   tensorboard --logdir benchmark_results/run_TIMESTAMP/dvsgesture/tensorboard
   ```

---

## 🎉 Expected Output Structure

After running benchmark and analysis:

```
benchmark_results/
└── run_20250114_143022/
    ├── benchmark_summary.json       # ← Main results file
    ├── BENCHMARK_REPORT.md          # ← Read this!
    ├── training_curves.png          # ← View this!
    ├── comparison.png               # ← View this!
    ├── dvsgesture_log.txt
    ├── cifar10dvs_log.txt
    ├── dvsgesture/
    │   ├── best_model.pth           # ← Best trained model
    │   ├── config.json
    │   └── tensorboard/
    └── cifar10dvs/
        ├── best_model.pth           # ← Best trained model
        ├── config.json
        └── tensorboard/
```

Start here: Open `BENCHMARK_REPORT.md` to see everything! 📊
