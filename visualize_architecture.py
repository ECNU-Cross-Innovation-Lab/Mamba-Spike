"""
Mamba-Spike架构可视化脚本
使用graphviz生成详细的模型架构图
"""

import torch
from models.mamba_spike import create_mamba_spike_nmnist
from torchviz import make_dot


def print_model_structure():
    """打印模型结构的文本版本"""
    print("=" * 80)
    print("Mamba-Spike Model Architecture")
    print("=" * 80)
    print()

    model = create_mamba_spike_nmnist()

    print("Model Summary")
    print("-" * 80)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += num_params
        trainable_params += num_trainable

        print(f"{name:30s} | Params: {num_params:>10,} | Trainable: {num_trainable:>10,}")

    print("-" * 80)
    print(f"{'Total':30s} | Params: {total_params:>10,} | Trainable: {trainable_params:>10,}")
    print("=" * 80)
    print()


def print_layer_details():
    """打印每一层的详细信息"""
    print("=" * 80)
    print("Detailed Layer Information")
    print("=" * 80)
    print()

    model = create_mamba_spike_nmnist()

    print("Layer-by-Layer Breakdown")
    print("-" * 80)

    for idx, (name, module) in enumerate(model.named_modules()):
        if len(list(module.children())) == 0:  # 只显示叶子节点
            module_name = type(module).__name__
            print(f"[{idx:3d}] {name:50s} | {module_name:20s}")

            # 显示参数形状
            for param_name, param in module.named_parameters(recurse=False):
                print(f"      └─ {param_name:30s}: {tuple(param.shape)}")

    print("=" * 80)
    print()


def visualize_data_flow():
    """可视化数据流和维度变化"""
    print("=" * 80)
    print("Data Flow Visualization")
    print("=" * 80)
    print()

    # 创建示例输入
    batch_size = 2
    time_steps = 10
    channels = 2
    height = 34
    width = 34

    x = torch.randn(batch_size, time_steps, channels, height, width)

    model = create_mamba_spike_nmnist()
    model.eval()

    print("Dimension Changes Through Network")
    print("-" * 80)
    print(f"{'Layer':<40s} | {'Output Shape':<30s}")
    print("-" * 80)

    # Input
    print(f"{'Input (Event Frames)':<40s} | {str(tuple(x.shape)):<30s}")

    # Spiking Frontend
    with torch.no_grad():
        spikes, mem = model.spiking_frontend(x)
        print(f"{'Spiking Frontend (Spikes)':<40s} | {str(tuple(spikes.shape)):<30s}")

        # Interface Layer
        activations = model.spike_to_activation(spikes)
        print(f"{'Interface Layer (Activations)':<40s} | {str(tuple(activations.shape)):<30s}")

        # Input Projection
        projected = model.input_proj(activations)
        print(f"{'Input Projection':<40s} | {str(tuple(projected.shape)):<30s}")

        # Mamba Blocks
        x_mamba = projected
        for i, block in enumerate(model.mamba_blocks):
            x_mamba = block(x_mamba)
            print(f"{f'Mamba Block {i+1}':<40s} | {str(tuple(x_mamba.shape)):<30s}")

        # Global Pooling
        pooled = x_mamba.mean(dim=1)
        print(f"{'Global Average Pooling':<40s} | {str(tuple(pooled.shape)):<30s}")

        # Normalization
        normed = model.norm(pooled)
        print(f"{'LayerNorm':<40s} | {str(tuple(normed.shape)):<30s}")

        # Classification
        logits = model.classifier(normed)
        print(f"{'Classifier (Output Logits)':<40s} | {str(tuple(logits.shape)):<30s}")

    print("=" * 80)
    print()


def analyze_parameters():
    """分析参数分布"""
    print("=" * 80)
    print("Parameter Analysis")
    print("=" * 80)
    print()

    model = create_mamba_spike_nmnist()

    print("Parameter Distribution by Component")
    print("-" * 80)

    components = {
        'Spiking Frontend': 'spiking_frontend',
        'Interface Layer': 'spike_to_activation',
        'Input Projection': 'input_proj',
        'Mamba Blocks': 'mamba_blocks',
        'Classification Head': ['norm', 'classifier']
    }

    total_params = sum(p.numel() for p in model.parameters())

    for comp_name, comp_key in components.items():
        if isinstance(comp_key, list):
            params = sum(
                sum(p.numel() for p in getattr(model, key).parameters())
                for key in comp_key
            )
        else:
            params = sum(p.numel() for p in getattr(model, comp_key).parameters())

        percentage = (params / total_params) * 100
        print(f"{comp_name:25s} | {params:>10,} | {percentage:>6.2f}%")

    print("-" * 80)
    print(f"{'Total Parameters':25s} | {total_params:>10,} | {100.00:>6.2f}%")
    print("=" * 80)
    print()


def print_ascii_architecture():
    """打印ASCII艺术风格的架构图"""
    print("=" * 80)
    print("ASCII Architecture Diagram")
    print("=" * 80)
    print()

    diagram = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    Event-based Input                        │
    │                  (B, T, 2, 34, 34)                         │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Spiking Front-End (SNN)                        │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(2→32) + MaxPool + LIF(β=0.9)                  │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │                           ▼                                 │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(32→32) + MaxPool + LIF(β=0.9)                 │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │                           ▼                                 │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(32→64) + LIF(β=0.9)                           │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                   Sparse Spikes                             │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Interface Layer                                │
    │    Spike → Activation Conversion                           │
    │    Rate Coding + Temporal Smoothing                        │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │           Input Projection                                  │
    │         Linear(features → 128)                             │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Mamba Backbone (SSM)                           │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 1: LayerNorm + SelectiveSSM + Residual   │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 2: LayerNorm + SelectiveSSM + Residual   │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 3: LayerNorm + SelectiveSSM + Residual   │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 4: LayerNorm + SelectiveSSM + Residual   │  │
    │  └───────────────────────────────────────────────────────┘  │
    │              Linear-Time Sequence Modeling                  │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │        Global Average Pooling (over time)                   │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │           Classification Head                               │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ LayerNorm(128)                                        │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Linear(128 → num_classes)                            │  │
    │  └───────────────────────────────────────────────────────┘  │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  Output Logits                              │
    │                 (B, num_classes)                            │
    └─────────────────────────────────────────────────────────────┘
    """

    print(diagram)
    print("=" * 80)
    print()


def main():
    """主函数"""
    print("\n")
    print("Mamba-Spike Architecture Visualization")
    print("=" * 80)
    print()

    # 1. ASCII架构图
    print_ascii_architecture()

    # 2. 模型结构总结
    print_model_structure()

    # 3. 数据流可视化
    visualize_data_flow()

    # 4. 参数分析
    analyze_parameters()

    # 5. 详细层信息（可选，信息量大）
    user_input = input("\n显示详细的层信息吗? (y/n): ")
    if user_input.lower() == 'y':
        print_layer_details()

    print("\n" + "=" * 80)
    print("可视化完成!")
    print("=" * 80)
    print()
    print("提示:")
    print("  1. 使用 draw.io 打开 architecture/mamba_spike_architecture.drawio")
    print("  2. 查看 architecture/README.md 了解更多细节")
    print("  3. 参考 models/mamba_spike.py 查看代码实现")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
