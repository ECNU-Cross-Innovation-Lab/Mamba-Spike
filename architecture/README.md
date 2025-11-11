# Mamba-Spike 架构图说明

## 📊 如何查看架构图

### 方法1: 在线查看（推荐）
1. 访问 [draw.io](https://app.diagrams.net/)
2. 点击 "Open Existing Diagram"
3. 上传 `mamba_spike_architecture.drawio` 文件
4. 即可查看和编辑

### 方法2: VS Code插件
```bash
# 安装Draw.io Integration插件
code --install-extension hediet.vscode-drawio

# 直接在VS Code中打开.drawio文件
```

### 方法3: 桌面应用
1. 下载 [diagrams.net Desktop](https://github.com/jgraph/drawio-desktop/releases)
2. 安装后打开 `.drawio` 文件

## 🏗️ 架构概览

### 完整流程

```
Event Input (DVS Camera)
    ↓
Spiking Front-End (LIF Neurons)
    ├─ Conv2d + LIF (32 channels)
    ├─ Conv2d + LIF (32 channels)
    └─ Conv2d + LIF (64 channels)
    ↓
Sparse Spikes (B, T, 64, H', W')
    ↓
Interface Layer (Spike-to-Activation)
    └─ Rate Coding + Temporal Smoothing
    ↓
Input Projection (Linear)
    └─ spike_features → 128
    ↓
Mamba Backbone (4 Layers)
    ├─ Mamba Block 1 (LayerNorm + SSM + Residual)
    ├─ Mamba Block 2
    ├─ Mamba Block 3
    └─ Mamba Block 4
    ↓
Global Average Pooling (over time)
    ↓
Classification Head
    ├─ LayerNorm
    └─ Linear (128 → num_classes)
    ↓
Output Logits (B, num_classes)
```

## 📐 架构细节

### 1. Spiking Front-End

**输入**: `(B, T, C, H, W)` - Event frames
- B: Batch size
- T: Time steps
- C: Channels (2 for DVS: ON/OFF)
- H, W: Height, Width

**结构**:
```python
Conv2d(2→32, kernel=3) + MaxPool2d(2) + LIF(β=0.9)
    ↓
Conv2d(32→32, kernel=3) + MaxPool2d(2) + LIF(β=0.9)
    ↓
Conv2d(32→64, kernel=3) + LIF(β=0.9)
```

**输出**: `(B, T, 64, H/4, W/4)` - Sparse spikes

**关键特性**:
- 🔥 LIF神经元: 生物学合理的脉冲机制
- ⚡ 事件驱动: 仅在必要时产生脉冲
- 💾 稀疏性: 显著降低计算和内存需求

### 2. Interface Layer

**功能**: 将离散脉冲转换为连续激活

**方法**:
```python
# Rate Coding
activations = spikes.view(B, T, -1)  # Flatten spatial

# Temporal Smoothing (Conv1d)
kernel = ones(features, 1, 5) / 5  # Moving average
activations = conv1d(activations, kernel, groups=features)
```

**输出**: `(B, T, 64×H'×W')` - Continuous activations

### 3. Mamba Backbone

**核心组件**: Selective SSM (State Space Model)

```python
class SelectiveSSM:
    def forward(x):
        # 1. Input projection
        x_proj = in_proj(x)  # Split into x and residual

        # 2. Convolution (local context)
        x = conv1d(x, kernel=4)

        # 3. State Space Model
        A = -exp(A_log)  # State transition
        dt, B, C = compute_parameters(x)  # Data-dependent!

        # 4. Selective Scan
        for i in range(L):
            h[i] = exp(dt[i]*A) * h[i-1] + dt[i]*B[i]*x[i]
            y[i] = C[i] * h[i] + D * x[i]

        # 5. Gated MLP
        y = y * silu(residual)

        return out_proj(y)
```

**关键创新**:
- 🎯 **选择性**: 参数dt, B, C依赖于输入
- ⚡ **线性复杂度**: O(L) vs Transformer的O(L²)
- 🧠 **长程依赖**: 有效建模长序列

**每个Mamba Block**:
```python
MambaBlock(x):
    return x + SSM(LayerNorm(x))  # Residual connection
```

### 4. Classification Head

```python
# Global pooling
x = x.mean(dim=1)  # (B, T, 128) → (B, 128)

# Normalization + Classification
x = LayerNorm(x)
logits = Linear(x)  # (B, 128) → (B, num_classes)
```

## 📊 参数统计

### 模型大小

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Spiking Front-End | ~20K | 1.6% |
| Interface + Projection | ~16K | 1.3% |
| Mamba Backbone (4层) | ~1.1M | 91.7% |
| Classification Head | ~65K | 5.4% |
| **总计** | **~1.2M** | **100%** |

### 不同数据集配置

| 数据集 | 输入尺寸 | Spiking Channels | d_model | n_layers | 参数量 |
|--------|---------|------------------|---------|----------|--------|
| N-MNIST | 34×34 | 64 | 128 | 4 | 1.2M |
| DVS Gesture | 128×128 | 128 | 256 | 6 | 8.5M |
| CIFAR10-DVS | 128×128 | 128 | 256 | 6 | 8.5M |

## 🔄 数据流分析

### 维度变化追踪

```
输入事件流: (32, 300, 2, 34, 34)
    ↓ [Conv+Pool+LIF]
稀疏脉冲: (32, 300, 64, 8, 8)
    ↓ [Flatten + Smooth]
激活向量: (32, 300, 4096)
    ↓ [Linear Projection]
嵌入序列: (32, 300, 128)
    ↓ [4× Mamba Blocks]
处理序列: (32, 300, 128)
    ↓ [Global Pooling]
聚合特征: (32, 128)
    ↓ [LayerNorm + Linear]
输出logits: (32, 10)
```

### 计算复杂度

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Spiking Front-End | O(T×H×W) | O(B×T×C×H×W) |
| Interface Layer | O(T×F) | O(B×T×F) |
| Mamba Backbone | **O(T×d²)** | O(B×T×d) |
| Classification | O(d×C) | O(B×d) |

**注意**: Mamba的线性时间复杂度相比Transformer的O(T²×d)有显著优势！

## 🎨 架构特点

### 1. 混合范式
- **前端**: 脉冲神经网络 (Event-driven)
- **主干**: 状态空间模型 (Continuous)
- **优势**: 结合两者优点

### 2. 稀疏性
- **空间稀疏**: 事件相机只记录变化
- **时间稀疏**: LIF神经元只在必要时发放
- **结果**: ~78.5% 稀疏率 (DVS Gesture)

### 3. 效率
- **计算**: 线性时间复杂度
- **能耗**: 稀疏脉冲降低功耗
- **内存**: 渐进式处理，无需全局注意力

### 4. 性能
| 数据集 | Mamba-Spike | 基准SNN | 基准ANN |
|--------|-------------|---------|---------|
| N-MNIST | **99.5%** | 98.8% | 99.2% |
| DVS Gesture | **97.8%** | 96.5% | 97.1% |
| CIFAR10-DVS | **92.5%** | 89.6% | 91.8% |

## 🔬 关键创新点

### 1. Selective State Spaces
```python
# 传统SSM: 参数固定
A, B, C = learnable_parameters()

# Selective SSM: 参数动态（依赖输入）
dt, B, C = f(x)  # 根据输入计算！
A = -exp(A_log)
```

**为什么重要?**
- 可以根据内容选择性地记住或忘记信息
- 类似注意力机制但更高效

### 2. Spike-to-Activation Interface
```python
# 保持时间信息
# 将稀疏脉冲→平滑激活
# 允许梯度反向传播
```

**为什么重要?**
- 桥接离散SNN和连续Mamba
- 保留脉冲的时序结构
- 允许端到端训练

## 📚 代码对应关系

### 架构图 ↔ 代码

```python
# models/mamba_spike.py

class MambaSpike(nn.Module):
    def __init__(...):
        # 对应图中的 "Spiking Front-End"
        self.spiking_frontend = SpikingFrontEnd(...)

        # 对应图中的 "Interface Layer"
        self.spike_to_activation = SpikeToActivation(...)

        # 对应图中的 "Input Projection"
        self.input_proj = nn.Linear(spike_features, d_model)

        # 对应图中的 "Mamba Backbone"
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(...) for _ in range(n_layers)
        ])

        # 对应图中的 "Classification Head"
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 按照架构图的流程
        spikes, _ = self.spiking_frontend(x)
        activations = self.spike_to_activation(spikes)
        x = self.input_proj(activations)

        for block in self.mamba_blocks:
            x = block(x)

        x = x.mean(dim=1)  # Global pooling
        x = self.norm(x)
        logits = self.classifier(x)

        return logits
```

## 🎯 使用建议

### 查看架构图时：
1. **从上到下**: 跟随数据流
2. **注意颜色**: 不同颜色代表不同模块类型
3. **查看Legend**: 了解各组件含义
4. **读Key Features**: 理解核心创新

### 修改架构图：
1. 在draw.io中打开
2. 可以调整布局、颜色、文字
3. 添加自己的注释
4. 导出为PNG/PDF/SVG

## 📖 相关文档

- `models/mamba_spike.py` - 模型实现代码
- `README.md` - 项目完整说明
- 论文: [arXiv:2408.11823](https://arxiv.org/abs/2408.11823)

## 🔗 在线资源

- [Draw.io 官网](https://app.diagrams.net/)
- [Mamba 论文](https://arxiv.org/abs/2312.00752)
- [snnTorch 文档](https://snntorch.readthedocs.io/)
