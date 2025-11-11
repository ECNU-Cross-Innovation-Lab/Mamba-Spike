"""
云端GPU训练脚本 - 针对NVIDIA GPU (RTX 3090/4090/5090, A100等) 优化
支持CUDA加速，性能远超本地训练
"""

import argparse
import sys
import torch

def check_cuda():
    """检查CUDA环境"""
    print("=" * 70)
    print("云端GPU训练配置检测")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA不可用!")
        print("\n可能原因:")
        print("  1. 没有NVIDIA GPU")
        print("  2. CUDA驱动未安装")
        print("  3. PyTorch未安装CUDA版本")
        print("\n解决方案:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    print(f"CUDA可用")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"   显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"   计算能力: {props.major}.{props.minor}")
        print(f"   多处理器: {props.multi_processor_count}")

    print("=" * 70)
    return True


def get_optimized_config(gpu_name, dataset):
    """根据GPU型号和数据集返回优化配置"""

    # GPU配置映射
    gpu_configs = {
        'RTX 5090': {'batch_size': 128, 'workers': 8, 'note': '顶级性能'},
        'RTX 4090': {'batch_size': 96, 'workers': 8, 'note': '极高性能'},
        'RTX 3090': {'batch_size': 64, 'workers': 6, 'note': '高性能'},
        'A100': {'batch_size': 128, 'workers': 8, 'note': '专业级'},
        'V100': {'batch_size': 64, 'workers': 6, 'note': '高性能'},
        'T4': {'batch_size': 32, 'workers': 4, 'note': '良好性能'},
    }

    # 检测GPU型号
    config = {'batch_size': 32, 'workers': 4, 'note': '标准配置'}

    for key in gpu_configs:
        if key.lower() in gpu_name.lower():
            config = gpu_configs[key]
            break

    return config


def estimate_training_time(gpu_name, dataset, epochs):
    """估算训练时间"""

    # 时间估算（分钟/100 epochs）
    time_estimates = {
        'RTX 5090': {'nmnist': 15, 'dvsgesture': 30, 'cifar10dvs': 45},
        'RTX 4090': {'nmnist': 20, 'dvsgesture': 40, 'cifar10dvs': 60},
        'RTX 3090': {'nmnist': 35, 'dvsgesture': 70, 'cifar10dvs': 90},
        'A100': {'nmnist': 18, 'dvsgesture': 35, 'cifar10dvs': 50},
        'V100': {'nmnist': 40, 'dvsgesture': 80, 'cifar10dvs': 100},
        'T4': {'nmnist': 90, 'dvsgesture': 180, 'cifar10dvs': 240},
    }

    # 查找匹配的GPU
    base_time = 120  # 默认时间（分钟）
    for key in time_estimates:
        if key.lower() in gpu_name.lower():
            if dataset in time_estimates[key]:
                base_time = time_estimates[key][dataset]
            break

    # 根据epochs调整
    estimated_minutes = base_time * (epochs / 100)

    if estimated_minutes < 60:
        return f"{estimated_minutes:.0f} 分钟"
    else:
        hours = estimated_minutes / 60
        return f"{hours:.1f} 小时"


def main():
    """云端GPU训练主函数"""

    # 检查CUDA
    check_cuda()

    parser = argparse.ArgumentParser(description='云端GPU训练 - NVIDIA CUDA加速')

    # Dataset
    parser.add_argument('--dataset', type=str, default='nmnist',
                        choices=['nmnist', 'dvsgesture', 'cifar10dvs'],
                        help='数据集选择')

    # Training parameters (针对高性能GPU优化)
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（默认100）')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小（留空自动选择）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='数据加载线程数（留空自动选择）')

    # Dataset parameters
    parser.add_argument('--time-window', type=int, default=300000,
                        help='时间窗口（微秒）')
    parser.add_argument('--dt', type=float, default=1000,
                        help='时间分辨率（微秒）')

    # System
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name(0)

    # 自动配置最优参数
    config = get_optimized_config(gpu_name, args.dataset)

    if args.batch_size is None:
        args.batch_size = config['batch_size']

    if args.num_workers is None:
        args.num_workers = config['workers']

    # 显示配置信息
    print("\n云端GPU训练配置")
    print("=" * 70)
    print(f"  GPU: {gpu_name} {config['note']}")
    print(f"  数据集: {args.dataset}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size} (GPU优化)")
    print(f"  学习率: {args.lr}")
    print(f"  数据加载线程: {args.num_workers}")
    print("=" * 70)

    # 估算训练时间
    est_time = estimate_training_time(gpu_name, args.dataset, args.epochs)
    print(f"\n 预计训练时间: {est_time}")
    print("=" * 70)

    # 性能提示
    print("\n性能优化提示:")
    print("  使用大batch size充分利用GPU")
    print("  启用混合精度训练（如果支持）")
    print("  使用多线程数据加载")
    print("  确保数据在SSD上以加快加载速度")
    print("=" * 70)

    print("\n开始训练...")
    print("=" * 70)

    # 导入训练模块
    from train import Trainer
    import numpy as np

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 创建训练器
    trainer = Trainer(args)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n 训练被用户中断")
        print(f"已保存的checkpoint可以在 {args.output_dir} 找到")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 显示后续步骤
    print("\n后续步骤:")
    print(f"1. 查看TensorBoard: tensorboard --logdir={args.output_dir}")
    print(f"2. 评估模型: python evaluate.py --checkpoint {args.output_dir}/.../checkpoint_best.pth")
    print(f"3. 下载结果: 打包outputs目录")
    print("=" * 70)


if __name__ == "__main__":
    main()
