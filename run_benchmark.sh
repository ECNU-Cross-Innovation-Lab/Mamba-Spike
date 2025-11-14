#!/bin/bash
# Quick benchmark script for Linux/Mac
# Runs 4 epochs on DVS Gesture and CIFAR10-DVS

echo "===================================================================="
echo "Mamba-Spike Benchmark - Quick Test"
echo "===================================================================="
echo ""
echo "This will train on:"
echo "  - DVS Gesture (4 epochs)"
echo "  - CIFAR10-DVS (4 epochs)"
echo ""
echo "Estimated time: 20-40 minutes (depends on GPU)"
echo ""
read -p "Press Enter to start..."

# Run benchmark
python run_benchmark.py --epochs 4 --batch-size-dvsgesture 16 --batch-size-cifar10dvs 16

echo ""
echo "===================================================================="
echo "Benchmark Complete!"
echo "===================================================================="
echo ""
echo "To analyze results, find the latest run directory:"
echo "  ls -lt benchmark_results/"
echo ""
echo "Then run:"
echo "  python analyze_benchmark.py benchmark_results/run_TIMESTAMP/"
echo ""
