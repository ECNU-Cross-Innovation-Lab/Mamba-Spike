# 云端GPU训练指南

如果你的MacBook训练速度较慢，可以使用云端GPU服务进行加速训练。

## 🚀 速度对比

| 设备 | 相对速度 | 100 epochs预估时间 |
|------|----------|-------------------|
| **MacBook M3 Max (MPS)** | 1x | 4-6 小时 |
| **Google Colab (T4 GPU)** | 3-5x | 1-2 小时 |
| **NVIDIA RTX 3090** | 8-10x | 30-45 分钟 |
| **NVIDIA A100** | 15-20x | 15-20 分钟 |

## 方案对比

### 方案1: Google Colab (推荐 - 免费)

**优点**:
- ✅ 完全免费（有使用时间限制）
- ✅ 提供Tesla T4 GPU
- ✅ 无需配置环境
- ✅ Jupyter Notebook界面

**缺点**:
- ⚠️ 会话有时间限制（最长12小时）
- ⚠️ 可能需要排队

**如何使用**:

1. 访问 [Google Colab](https://colab.research.google.com/)
2. 新建notebook
3. 启用GPU: `运行时` → `更改运行时类型` → `GPU`
4. 运行以下代码：

```python
# 克隆项目
!git clone https://github.com/ECNU-Cross-Innovation-Lab/Mamba-Spike.git
%cd Mamba-Spike

# 安装依赖
!pip install -q torch torchvision torchaudio
!pip install -q tonic snntorch einops tensorboard

# 检查GPU
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 开始训练
!python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir ./outputs
```

### 方案2: Google Colab Pro (付费)

**价格**: $9.99/月

**优点**:
- ✅ 更快的GPU (V100/A100)
- ✅ 更长的会话时间
- ✅ 优先访问

**推荐**: 如果经常训练大模型

### 方案3: Kaggle Notebooks (免费)

**优点**:
- ✅ 每周30小时GPU时间
- ✅ NVIDIA Tesla P100 GPU
- ✅ 稳定可靠

**如何使用**:

1. 访问 [Kaggle](https://www.kaggle.com/)
2. 创建新Notebook
3. 设置: `Accelerator` → `GPU`
4. 上传代码或从GitHub导入

### 方案4: AWS SageMaker (按需付费)

**优点**:
- ✅ 多种GPU选择
- ✅ 可扩展性强
- ✅ 适合大规模训练

**成本**:
- ml.g4dn.xlarge (T4 GPU): ~$0.50/小时
- ml.p3.2xlarge (V100 GPU): ~$3.00/小时

### 方案5: Paperspace Gradient (按需付费)

**优点**:
- ✅ 简单易用
- ✅ 价格合理
- ✅ Jupyter环境

**成本**:
- Free GPU: 免费但资源有限
- P4000: $0.45/小时
- RTX4000: $0.56/小时

## 📝 Google Colab 完整训练脚本

创建新的Colab Notebook，复制以下代码：

```python
# ===== Cell 1: 环境准备 =====
# 克隆代码
!git clone https://github.com/ECNU-Cross-Innovation-Lab/Mamba-Spike.git
%cd Mamba-Spike

# 安装依赖
!pip install -q torch torchvision torchaudio
!pip install -q tonic snntorch einops tensorboard pandas scikit-learn seaborn matplotlib tqdm

# 检查GPU
import torch
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ===== Cell 2: 训练N-MNIST =====
# 训练模型（使用CUDA加速）
!python train.py \
    --dataset nmnist \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --clip-grad 1.0 \
    --num-workers 2 \
    --output-dir ./outputs

# ===== Cell 3: 下载训练结果 =====
# 打包输出文件
!zip -r outputs.zip outputs/

# 下载到本地
from google.colab import files
files.download('outputs.zip')

# ===== Cell 4: 可视化训练曲线 =====
# 加载TensorBoard
%load_ext tensorboard
%tensorboard --logdir outputs/
```

## 💡 MacBook M3 Max 本地优化建议

虽然不如CUDA快，但M3 Max的MPS加速仍然比CPU快很多。优化建议：

### 1. 使用优化配置

```bash
# 使用为M3优化的脚本
chmod +x configs/m3_optimized.sh
./configs/m3_optimized.sh
```

### 2. 调整参数

```bash
# 减小batch size（降低内存压力）
python train.py --dataset nmnist --batch-size 16

# 减少workers（避免CPU瓶颈）
python train.py --dataset nmnist --num-workers 2

# 减少time bins（减少计算量）
python train.py --dataset nmnist --dt 2000  # 2ms instead of 1ms
```

### 3. 先用小规模测试

```bash
# 只训练10个epochs看效果
python train.py --dataset nmnist --epochs 10 --batch-size 16

# 如果满意再运行完整训练
python train.py --dataset nmnist --epochs 100 --batch-size 16
```

### 4. 后台训练

```bash
# 使用nohup后台运行，不怕断线
nohup python train.py --dataset nmnist --epochs 100 > training.log 2>&1 &

# 查看进度
tail -f training.log
```

### 5. 节能模式关闭

```
系统设置 → 电池 → 关闭"低电量模式"
系统设置 → 电池 → 打开"阻止Mac自动进入睡眠"
```

## 🎯 推荐方案

根据你的情况：

| 场景 | 推荐方案 |
|------|----------|
| **快速测试** | MacBook M3 Max (本地) |
| **完整训练** | Google Colab (免费GPU) |
| **频繁训练** | Colab Pro ($9.99/月) |
| **生产环境** | AWS/Azure (按需) |

## 📊 预估训练时间

### N-MNIST (100 epochs)

- **MacBook M3 Max**: 4-6 小时
- **Google Colab T4**: 1-2 小时
- **RTX 3090**: 30-45 分钟
- **A100**: 15-20 分钟

### DVS Gesture (150 epochs)

- **MacBook M3 Max**: 8-12 小时
- **Google Colab T4**: 2-4 小时
- **RTX 3090**: 1-1.5 小时

### CIFAR10-DVS (200 epochs)

- **MacBook M3 Max**: 12-18 小时
- **Google Colab T4**: 4-6 小时
- **RTX 3090**: 1.5-2 小时

## 🔄 混合方案（推荐）

1. **本地开发和调试** (M3 Max)
   - 运行 `test_setup.py`
   - 运行 `simple_train.py`
   - 快速测试代码修改

2. **云端完整训练** (Colab)
   - 完整的100+ epochs训练
   - 获得最佳性能
   - 节省本地电量和时间

3. **本地评估分析** (M3 Max)
   - 下载训练好的模型
   - 运行评估和可视化
   - 分析结果

## 🆘 常见问题

**Q: Colab会话断开怎么办？**
A: 代码会自动保存checkpoint，重新连接后继续训练：
```python
!python train.py --resume outputs/nmnist_*/checkpoint_latest.pth
```

**Q: M3 Max训练会不会损坏电脑？**
A: 不会。M3 Max设计用于高负载，有完善的温控系统。

**Q: 如何监控M3 Max温度？**
A: 使用 Activity Monitor 或 iStat Menus

**Q: MPS vs CUDA性能差距大吗？**
A: 对于SNN，MPS大约是T4 GPU的 30-40%速度，仍比CPU快5-10倍。

## 📞 需要帮助？

- GitHub Issues: 报告问题
- 论文作者联系方式: 见论文
- 社区讨论: 待建立

---

**建议**: 如果你只是想快速体验，用M3 Max训练10-20个epochs就够了。如果要复现论文结果，建议使用Google Colab的免费GPU。
