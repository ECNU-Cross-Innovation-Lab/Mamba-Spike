#!/bin/bash
# MacBook M3 Max 优化配置
# 针对Apple Silicon优化的训练脚本

echo "=========================================="
echo "MacBook M3 Max 优化训练配置"
echo "=========================================="

# 检查MPS是否可用
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS加速可用')
else:
    print('❌ MPS加速不可用')
    exit(1)
"

# N-MNIST 优化配置
# 使用较小的batch size以适应内存
# 减少workers避免过多的CPU开销
python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --clip-grad 1.0 \
    --num-workers 2 \
    --output-dir ./outputs/m3_nmnist

echo "=========================================="
echo "训练完成！"
echo "=========================================="
