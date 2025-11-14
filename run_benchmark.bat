@echo off
REM Quick benchmark script for Windows
REM Runs 4 epochs on DVS Gesture and CIFAR10-DVS

echo ====================================================================
echo Mamba-Spike Benchmark - Quick Test
echo ====================================================================
echo.
echo This will train on:
echo   - DVS Gesture (4 epochs)
echo   - CIFAR10-DVS (4 epochs)
echo.
echo Estimated time: 20-40 minutes (depends on GPU)
echo.
pause

REM Run benchmark
python run_benchmark.py --epochs 4 --batch-size-dvsgesture 16 --batch-size-cifar10dvs 16

echo.
echo ====================================================================
echo Benchmark Complete!
echo ====================================================================
echo.
echo To analyze results, run:
echo   python analyze_benchmark.py benchmark_results\run_TIMESTAMP\
echo.
echo Replace TIMESTAMP with the actual folder name in benchmark_results\
echo.
pause
