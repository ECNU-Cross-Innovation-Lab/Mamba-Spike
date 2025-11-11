# Mamba-Spike 快速开始指南

## 📋 前置要求

- Python 3.9 或更高版本
- CUDA 11.0+ (如果使用GPU)
- 至少 8GB RAM
- 至少 10GB 可用磁盘空间 (用于数据集)

## 🔧 步骤1: 克隆仓库

```bash
# 克隆项目
git clone https://github.com/ECNU-Cross-Innovation-Lab/Mamba-Spike.git
cd Mamba-Spike
```

## 🐍 步骤2: 创建虚拟环境 (推荐)

### 使用 conda (推荐):
```bash
# 创建新环境
conda create -n mambaspike python=3.9

# 激活环境
conda activate mambaspike
```

### 使用 venv:
```bash
# 创建虚拟环境
python -m venv venv

# 激活环境 (Linux/Mac)
source venv/bin/activate

# 激活环境 (Windows)
venv\Scripts\activate
```

## 📦 步骤3: 安装依赖

```bash
# 安装PyTorch (根据你的系统选择合适的版本)
# CUDA版本 (如果有GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本 (如果没有GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Mac (Apple Silicon):
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

## 🚀 步骤4: 快速测试

### 测试模型创建:
```bash
python -c "
from models.mamba_spike import create_mamba_spike_nmnist
import torch

model = create_mamba_spike_nmnist()
print(f'模型创建成功!')
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

# 测试前向传播
batch_size = 2
time_steps = 10
x = torch.randn(batch_size, time_steps, 2, 34, 34)
output = model(x)
print(f'输入形状: {x.shape}')
print(f'输出形状: {output.shape}')
print('✅ 模型测试通过!')
"
```

### 测试数据加载:
```bash
python -c "
from data.dataset_loader import prepare_nmnist_dataset

print('开始下载N-MNIST数据集...')
train_loader, test_loader, num_classes = prepare_nmnist_dataset(
    batch_size=4,
    num_workers=0  # 测试时使用0
)
print(f'✅ 数据集加载成功!')
print(f'类别数: {num_classes}')
print(f'训练批次数: {len(train_loader)}')
print(f'测试批次数: {len(test_loader)}')
"
```

## 🏋️ 步骤5: 训练模型

### 快速训练 (小规模测试):
```bash
# 在N-MNIST上训练5个epoch
python train.py \
    --dataset nmnist \
    --epochs 5 \
    --batch-size 32 \
    --lr 1e-3 \
    --output-dir ./outputs
```

### 完整训练 (N-MNIST):
```bash
# 完整训练100个epoch
python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --clip-grad 1.0 \
    --output-dir ./outputs
```

### 训练其他数据集:

**DVS Gesture:**
```bash
python train.py \
    --dataset dvsgesture \
    --epochs 150 \
    --batch-size 16 \
    --lr 5e-4 \
    --time-window 500000 \
    --output-dir ./outputs
```

**CIFAR10-DVS:**
```bash
python train.py \
    --dataset cifar10dvs \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-3 \
    --time-window 1000000 \
    --dt 10000 \
    --output-dir ./outputs
```

## 📊 步骤6: 监控训练

训练时会自动创建TensorBoard日志:

```bash
# 在另一个终端中启动TensorBoard
tensorboard --logdir=./outputs
```

然后在浏览器中访问: http://localhost:6006

## 🧪 步骤7: 评估模型

```bash
# 评估训练好的模型
python evaluate.py \
    --checkpoint outputs/nmnist_20240816_120000/checkpoint_best.pth \
    --dataset nmnist
```

## 💡 常见问题

### 1. CUDA Out of Memory
```bash
# 减小batch size
python train.py --dataset nmnist --batch-size 16  # 或更小
```

### 2. 数据加载慢
```bash
# 减少workers数量
python train.py --dataset nmnist --num-workers 2
```

### 3. 没有GPU
```bash
# 代码会自动使用CPU，但训练会较慢
# 建议使用较小的模型或减少epochs
python train.py --dataset nmnist --epochs 10 --batch-size 8
```

### 4. Mac (Apple Silicon) 使用MPS加速
```bash
# 代码会自动检测并使用MPS加速
# 确保PyTorch版本 >= 2.0
python train.py --dataset nmnist --batch-size 32
```

## 📁 输出文件结构

训练后会在 `outputs/` 目录生成:

```
outputs/
└── nmnist_20240816_120000/
    ├── config.json              # 训练配置
    ├── checkpoint_best.pth      # 最佳模型
    ├── checkpoint_latest.pth    # 最新模型
    └── tensorboard/             # TensorBoard日志
```

## 🎯 预期结果

根据论文，预期达到的准确率:

- **N-MNIST**: ~99.5%
- **DVS Gesture**: ~97.8%
- **CIFAR10-DVS**: ~92.5%

注意: 实际结果可能因硬件、随机种子等因素略有差异。

## 📝 自定义训练

### 修改超参数:
```bash
python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --clip-grad 1.0 \
    --time-window 300000 \
    --dt 1000 \
    --seed 42
```

### 查看所有参数:
```bash
python train.py --help
```

## 🔍 调试模式

如果遇到问题，可以使用小数据集快速测试:

```bash
# 只训练1个epoch，batch size设为2
python train.py \
    --dataset nmnist \
    --epochs 1 \
    --batch-size 2 \
    --num-workers 0
```

## 📚 更多信息

- 完整文档: [README.md](README.md)
- 论文: [arXiv:2408.11823](https://arxiv.org/abs/2408.11823)
- GitHub Issues: 报告问题和获取帮助
