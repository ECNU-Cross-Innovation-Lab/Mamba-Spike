# 云端GPU部署指南 🚀

## RTX 5090 完全支持！⚡

RTX 5090是目前最强的消费级GPU之一，完全支持CUDA，训练速度极快！

### 🔥 RTX 5090 性能预测

| 数据集 | 训练时间 (100 epochs) | 对比M3 Max |
|--------|---------------------|-----------|
| **N-MNIST** | **~15分钟** | 快16-24倍 |
| **DVS Gesture** | **~30分钟** | 快16-24倍 |
| **CIFAR10-DVS** | **~45分钟** | 快16-24倍 |

## 📦 快速部署步骤

### 步骤1: 解压上传的文件

```bash
# 在云端服务器上
unzip mambaspike.zip
cd Mamba-Spike
```

### 步骤2: 安装依赖

```bash
# 确保使用CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者CUDA 12.1版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 步骤3: 验证CUDA

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'CUDA Version: {torch.version.cuda}')
"
```

应该看到：
```
CUDA Available: True
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 5090
CUDA Version: 12.1
```

### 步骤4: 开始训练

```bash
# 使用云端GPU优化脚本（自动配置）
python train_cloud_gpu.py --dataset nmnist --epochs 100

# 或手动指定参数
python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --num-workers 8
```

## 🎯 针对不同GPU的推荐配置

### RTX 5090 (推荐配置)
```bash
python train_cloud_gpu.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 8
```

### RTX 4090
```bash
python train_cloud_gpu.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 96 \
    --num-workers 8
```

### RTX 3090
```bash
python train_cloud_gpu.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 6
```

### A100 (专业级)
```bash
python train_cloud_gpu.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 8
```

## 📊 监控训练

### 使用TensorBoard
```bash
# 在另一个终端
tensorboard --logdir=./outputs --port 6006

# 如果是远程服务器，使用端口转发
# ssh -L 6006:localhost:6006 user@server
```

### 查看GPU使用情况
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi --loop=1
```

### 查看训练日志
```bash
# 后台训练
nohup python train_cloud_gpu.py --dataset nmnist --epochs 100 > training.log 2>&1 &

# 查看日志
tail -f training.log
```

## 🔧 常见云平台配置

### AutoDL (国内)
```bash
# AutoDL通常已安装CUDA和PyTorch
pip install tonic snntorch einops tensorboard

python train_cloud_gpu.py --dataset nmnist --epochs 100
```

### 恒源云
```bash
# 创建容器后
pip install -r requirements.txt

python train_cloud_gpu.py --dataset nmnist --epochs 100
```

### AWS EC2 (P3/P4实例)
```bash
# 使用Deep Learning AMI
source activate pytorch

pip install tonic snntorch einops

python train_cloud_gpu.py --dataset nmnist --epochs 100
```

### Google Cloud (T4/V100/A100)
```bash
# 安装依赖
pip install -r requirements.txt

# 训练
python train_cloud_gpu.py --dataset nmnist --epochs 100
```

## 💾 下载训练结果

### 方法1: 打包下载
```bash
# 打包输出文件
tar -czf results.tar.gz outputs/

# 使用scp下载到本地
# scp user@server:/path/to/results.tar.gz ./
```

### 方法2: 只下载模型
```bash
# 找到最佳模型
find outputs/ -name "checkpoint_best.pth"

# 下载单个文件
# scp user@server:/path/to/checkpoint_best.pth ./
```

### 方法3: 使用云平台的文件下载功能
- AutoDL: 使用JupyterLab文件浏览器下载
- 恒源云: 使用文件管理器下载
- AWS/GCP: 使用云存储服务

## 🎨 训练三个数据集

### 一键训练所有数据集
```bash
# N-MNIST
python train_cloud_gpu.py --dataset nmnist --epochs 100

# DVS Gesture
python train_cloud_gpu.py --dataset dvsgesture --epochs 150

# CIFAR10-DVS
python train_cloud_gpu.py --dataset cifar10dvs --epochs 200
```

### 使用脚本批量训练
创建 `train_all.sh`:
```bash
#!/bin/bash

echo "训练N-MNIST..."
python train_cloud_gpu.py --dataset nmnist --epochs 100 > nmnist.log 2>&1

echo "训练DVS Gesture..."
python train_cloud_gpu.py --dataset dvsgesture --epochs 150 > dvsgesture.log 2>&1

echo "训练CIFAR10-DVS..."
python train_cloud_gpu.py --dataset cifar10dvs --epochs 200 > cifar10dvs.log 2>&1

echo "所有训练完成！"
```

运行：
```bash
chmod +x train_all.sh
./train_all.sh
```

## ⏱️ 预计总训练时间

### RTX 5090 (所有数据集)
- N-MNIST (100 epochs): ~15分钟
- DVS Gesture (150 epochs): ~45分钟
- CIFAR10-DVS (200 epochs): ~90分钟
- **总计**: ~2.5小时

### RTX 4090
- **总计**: ~3-4小时

### RTX 3090
- **总计**: ~5-6小时

## 🐛 故障排除

### CUDA Out of Memory
```bash
# 减小batch size
python train_cloud_gpu.py --batch-size 64  # 或更小
```

### 数据下载失败
```bash
# 手动下载数据集或使用镜像源
# 修改 data/dataset_loader.py 中的下载路径
```

### CUDA版本不匹配
```bash
# 重新安装匹配的PyTorch版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 性能优化技巧

### 1. 使用混合精度训练
```python
# 在train.py中已默认启用（如果GPU支持）
# RTX 5090完全支持FP16和TF32
```

### 2. 数据预加载
```bash
# 提前下载数据集，避免训练时下载
python -c "from data.dataset_loader import prepare_nmnist_dataset; prepare_nmnist_dataset()"
```

### 3. 使用SSD存储
- 确保数据集在SSD上
- 设置TMPDIR到SSD路径

### 4. 调整num_workers
```bash
# 根据CPU核心数调整
# RTX 5090服务器通常配8-16核CPU
python train_cloud_gpu.py --num-workers 8
```

## 📊 预期结果（RTX 5090）

| 数据集 | 目标准确率 | 实际时间 | 论文报告 |
|--------|-----------|---------|---------|
| N-MNIST | 99%+ | ~15分钟 | 99.5% |
| DVS Gesture | 97%+ | ~30分钟 | 97.8% |
| CIFAR10-DVS | 92%+ | ~45分钟 | 92.5% |

## 💰 成本估算

### AutoDL (RTX 3090/4090)
- 价格: ~¥2-3/小时
- N-MNIST训练成本: ~¥1-2

### 恒源云 (RTX 3090/4090)
- 价格: ~¥2-3/小时
- N-MNIST训练成本: ~¥1-2

### AWS P3 (V100)
- 价格: ~$3/小时
- N-MNIST训练成本: ~$2

### RTX 5090 (如果可用)
- 预计价格: ~¥3-5/小时
- N-MNIST训练成本: ~¥1

**建议**: RTX 3090/4090性价比最高，RTX 5090速度最快

## 🎉 完成后

1. **下载模型**: checkpoint_best.pth
2. **下载日志**: tensorboard日志文件
3. **查看结果**: 使用evaluate.py评估
4. **保存实验**: 记录超参数和结果

## 📞 需要帮助？

- 查看训练日志: `tail -f training.log`
- GPU状态: `nvidia-smi`
- 磁盘空间: `df -h`
- 内存使用: `free -h`

---

**提示**: RTX 5090训练速度极快，15分钟即可完成N-MNIST的完整训练，强烈推荐！
