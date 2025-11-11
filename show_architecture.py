"""
Mamba-Spike架构展示（无需安装依赖）
纯文本展示模型架构
"""


def print_ascii_architecture():
    """打印ASCII艺术风格的架构图"""
    print("=" * 80)
    print("Mamba-Spike Architecture Diagram")
    print("=" * 80)
    print()

    diagram = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    Event-based Input                        │
    │                  (B, T, 2, 34, 34)                         │
    │                  DVS Camera Events                          │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Spiking Front-End (SNN)                     │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(2→32, 3x3) + MaxPool(2x2) + LIF(β=0.9)        │  │
    │  │          ↓ Sparse Spikes                              │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │                           ▼                                 │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(32→32, 3x3) + MaxPool(2x2) + LIF(β=0.9)       │  │
    │  │          ↓ Sparse Spikes                              │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │                           ▼                                 │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Conv2d(32→64, 3x3) + LIF(β=0.9)                      │  │
    │  │          ↓ Sparse Spikes                              │  │
    │  └───────────────────────────────────────────────────────┘  │
    │         Output: (B, T, 64, H/4, W/4)                       │
    │         Feature: Event-driven, Energy-efficient            │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Interface Layer                             │
    │    Spike → Activation Conversion                           │
    │    • Rate Coding (temporal averaging)                      │
    │    • Temporal Smoothing (Conv1d, kernel=5)                │
    │         Output: (B, T, 64×H'×W')                          │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │           Input Projection                               │
    │         Linear(spike_features → 128)                       │
    │         Output: (B, T, 128)                                │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Mamba Backbone (SSM)                        │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 1                                      │  │
    │  │   ├─ LayerNorm(128)                                   │  │
    │  │   ├─ SelectiveSSM                                     │  │
    │  │   │   ├─ Input Projection (d_model → 2×d_inner)      │  │
    │  │   │   ├─ Conv1d (kernel=4, groups=d_inner)           │  │
    │  │   │   ├─ State Space Model                           │  │
    │  │   │   │   • A = -exp(A_log)                          │  │
    │  │   │   │   • dt, B, C = f(x)  [Data-dependent!]      │  │
    │  │   │   │   • h[t] = exp(dt*A)*h[t-1] + dt*B*x[t]     │  │
    │  │   │   │   • y[t] = C*h[t] + D*x[t]                  │  │
    │  │   │   └─ Gated MLP (y * silu(res))                   │  │
    │  │   └─ Residual Connection (+)                          │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 2 (same structure)                     │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 3 (same structure)                     │  │
    │  └────────────────────────┬──────────────────────────────┘  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Mamba Block 4 (same structure)                     │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                                                              │
    │    Feature: O(L) complexity, Long-range modeling            │
    │    Output: (B, T, 128)                                      │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │        Global Average Pooling                            │
    │           mean over time dimension                          │
    │           Output: (B, 128)                                  │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │           Classification Head                            │
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
    │                  Output Logits                           │
    │                 (B, num_classes)                            │
    │              Class Predictions                              │
    └─────────────────────────────────────────────────────────────┘
    """

    print(diagram)
    print("=" * 80)
    print()


def print_model_statistics():
    """打印模型统计信息"""
    print("=" * 80)
    print("Model Statistics")
    print("=" * 80)
    print()

    stats = """
    Parameter Distribution:
    ┌──────────────────────────┬──────────────┬──────────┐
    │ Component                │ Parameters   │ Ratio    │
    ├──────────────────────────┼──────────────┼──────────┤
    │ Spiking Front-End        │     ~20K     │   1.6%   │
    │ Interface + Projection   │     ~16K     │   1.3%   │
    │ Mamba Backbone (4 layers)│    ~1.1M     │  91.7%   │
    │ Classification Head      │     ~65K     │   5.4%   │
    ├──────────────────────────┼──────────────┼──────────┤
    │ Total                    │    ~1.2M     │  100%    │
    └──────────────────────────┴──────────────┴──────────┘

    Configurations for Different Datasets:
    ┌──────────────┬───────────┬─────────┬─────────┬────────┬──────────┐
    │ Dataset      │ Input     │ Spk_Ch  │ d_model │ Layers │ Params   │
    ├──────────────┼───────────┼─────────┼─────────┼────────┼──────────┤
    │ N-MNIST      │  34×34    │   64    │   128   │   4    │  ~1.2M   │
    │ DVS Gesture  │ 128×128   │  128    │   256   │   6    │  ~8.5M   │
    │ CIFAR10-DVS  │ 128×128   │  128    │   256   │   6    │  ~8.5M   │
    └──────────────┴───────────┴─────────┴─────────┴────────┴──────────┘

    Performance Results (from paper):
    ┌──────────────┬──────────────┬────────────┬────────────┐
    │ Dataset      │ Mamba-Spike  │ Baseline   │ Spikes/    │
    │              │ Accuracy     │ SNN        │ Sample     │
    ├──────────────┼──────────────┼────────────┼────────────┤
    │ N-MNIST      │   99.5%      │   98.8%    │    -       │
    │ DVS Gesture  │   97.8%      │   96.5%    │   785      │
    │ TIDIGITS     │   99.2%      │   98.3%    │    -       │
    │ CIFAR10-DVS  │   92.5%      │   89.6%    │    -       │
    └──────────────┴──────────────┴────────────┴────────────┘
    """

    print(stats)
    print("=" * 80)
    print()


def print_key_features():
    """打印关键特性"""
    print("=" * 80)
    print("Key Features & Innovations")
    print("=" * 80)
    print()

    features = """
    1. Event-Driven Spiking Front-End
       • Leaky Integrate-and-Fire (LIF) neurons
       • Biologically plausible spike generation
       • Energy-efficient sparse computation
       • Direct processing of DVS camera events

    2. Efficient Spike-to-Activation Interface
       • Rate coding with temporal averaging
       • Smooth conversion preserving temporal structure
       • Enables gradient backpropagation
       • Bridges discrete SNN and continuous Mamba

    3. Selective State Space Models (SSM)
       • Data-dependent parameters: dt, B, C = f(x)
       • Linear time complexity: O(L) vs O(L²) in Transformers
       • Long-range temporal dependencies
       • Selective information retention/forgetting

    4. Hybrid Architecture Benefits
       • Combines SNN efficiency with SSM performance
       • 78.5% sparsity on DVS Gesture dataset
       • Lower latency: 15ms vs 18-25ms (baselines)
       • Higher accuracy across all datasets

    5. Computational Advantages
       • Linear-time sequence processing
       • Reduced memory footprint
       • GPU-friendly parallel computation
       • Scalable to long sequences
    """

    print(features)
    print("=" * 80)
    print()


def print_data_flow():
    """打印数据流"""
    print("=" * 80)
    print("Data Flow Example (N-MNIST)")
    print("=" * 80)
    print()

    flow = """
    Step-by-Step Dimension Changes:

    Input:
    └─ Event Frames:           (32, 300, 2, 34, 34)
                                 ↓ [B, T, C, H, W]

    Spiking Front-End:
    ├─ After Conv1 + Pool:     (32, 300, 32, 17, 17)
    ├─ After Conv2 + Pool:     (32, 300, 32, 8, 8)
    └─ After Conv3:            (32, 300, 64, 8, 8)
                                 ↓ Sparse Spikes

    Interface Layer:
    ├─ Flatten spatial:        (32, 300, 4096)
    └─ Temporal smooth:        (32, 300, 4096)
                                 ↓ Continuous activations

    Input Projection:
    └─ Linear projection:      (32, 300, 128)
                                 ↓ Embedded sequence

    Mamba Backbone:
    ├─ After Block 1:          (32, 300, 128)
    ├─ After Block 2:          (32, 300, 128)
    ├─ After Block 3:          (32, 300, 128)
    └─ After Block 4:          (32, 300, 128)
                                 ↓ Temporal features

    Global Pooling:
    └─ Mean over time:         (32, 128)
                                 ↓ Aggregated features

    Classification:
    ├─ LayerNorm:              (32, 128)
    └─ Linear:                 (32, 10)
                                 ↓ Class logits

    Output:
    └─ Predictions:            (32, 10)
                                 [Batch, Classes]
    """

    print(flow)
    print("=" * 80)
    print()


def print_code_mapping():
    """打印代码映射"""
    print("=" * 80)
    print("Architecture → Code Mapping")
    print("=" * 80)
    print()

    mapping = """
    models/mamba_spike.py:

    class MambaSpike(nn.Module):

        # Spiking Front-End
        self.spiking_frontend = SpikingFrontEnd(
            in_channels=2,
            hidden_channels=32,
            out_channels=64,
            beta=0.9  # LIF decay rate
        )

        # Interface Layer
        self.spike_to_activation = SpikeToActivation(
            method="rate"  # Rate coding
        )

        # Input Projection
        self.input_proj = nn.Linear(
            spike_features,  # 64 * H' * W'
            d_model          # 128
        )

        # Mamba Backbone
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=128,
                d_state=16,
                d_conv=4,
                expand=2
            )
            for _ in range(4)  # 4 layers
        ])

        # Classification Head
        self.norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, num_classes)

    Forward Pass:
        spikes, _ = self.spiking_frontend(x)    # Event → Spikes
        activations = self.spike_to_activation(spikes)  # Spikes → Activation
        x = self.input_proj(activations)        # Project to d_model
        for block in self.mamba_blocks:         # Process with SSM
            x = block(x)
        x = x.mean(dim=1)                       # Global pooling
        x = self.norm(x)                        # Normalize
        logits = self.classifier(x)             # Classify
        return logits
    """

    print(mapping)
    print("=" * 80)
    print()


def main():
    """主函数"""
    print("\n")
    print("Mamba-Spike Architecture Visualization")
    print("=" * 80)
    print()

    print_ascii_architecture()
    print_model_statistics()
    print_key_features()
    print_data_flow()
    print_code_mapping()

    print("=" * 80)
    print("Additional Resources")
    print("=" * 80)
    print()
    print("1. Draw.io Diagram:")
    print("   Open 'architecture/mamba_spike_architecture.drawio'")
    print("   in https://app.diagrams.net/")
    print()
    print("2. Detailed Documentation:")
    print("   Read 'architecture/README.md'")
    print()
    print("3. Code Implementation:")
    print("   See 'models/mamba_spike.py'")
    print()
    print("4. Original Paper:")
    print("   https://arxiv.org/abs/2408.11823")
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
