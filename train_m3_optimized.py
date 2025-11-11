"""
M3 Max 优化训练脚本
专门针对Apple Silicon (M1/M2/M3) 优化的训练配置
"""

import argparse
import sys
import torch

# 检查MPS是否可用
if not torch.backends.mps.is_available():
    print("❌ MPS加速不可用!")
    print("可能原因:")
    print("  1. PyTorch版本过低 (需要 >= 2.0)")
    print("  2. macOS版本过低 (需要 >= 12.3)")
    print("  3. 不是Apple Silicon设备")
    sys.exit(1)

print("=" * 60)
print("M3 Max 优化训练配置")
print("=" * 60)
print(f"✅ MPS加速已启用")
print(f"✅ PyTorch版本: {torch.__version__}")
print("=" * 60)

# 导入训练模块
from train import Trainer, main as train_main

def main():
    """M3优化的训练配置"""

    parser = argparse.ArgumentParser(description='M3 Max优化训练')

    # Dataset
    parser.add_argument('--dataset', type=str, default='nmnist',
                        choices=['nmnist', 'dvsgesture', 'cifar10dvs'],
                        help='数据集选择')

    # 针对M3优化的默认参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数 (默认50，比完整训练少以节省时间)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小 (默认16，适合M3内存)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='数据加载线程数 (默认2，避免CPU瓶颈)')

    # 数据集参数
    parser.add_argument('--time-window', type=int, default=300000,
                        help='时间窗口 (微秒)')
    parser.add_argument('--dt', type=float, default=1000,
                        help='时间分辨率 (微秒)')

    # 系统参数
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 打印优化建议
    print("\n💡 M3 Max训练优化建议:")
    print("=" * 60)
    print("1. 关闭低电量模式以获得最佳性能")
    print("2. 连接电源适配器")
    print("3. 关闭其他占用内存的应用")
    print("4. 训练时不要让Mac进入睡眠")
    print("=" * 60)

    print("\n📊 训练配置:")
    print("=" * 60)
    print(f"  数据集: {args.dataset}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  数据加载线程: {args.num_workers}")
    print("=" * 60)

    # 预估训练时间
    estimated_times = {
        'nmnist': {
            50: '2-3',
            100: '4-6'
        },
        'dvsgesture': {
            50: '3-5',
            150: '8-12'
        },
        'cifar10dvs': {
            50: '4-6',
            200: '12-18'
        }
    }

    if args.dataset in estimated_times:
        if args.epochs in estimated_times[args.dataset]:
            est_time = estimated_times[args.dataset][args.epochs]
            print(f"\n⏱️  预计训练时间: {est_time} 小时")
        else:
            print(f"\n⏱️  预计训练时间: 根据epochs数量而定")

    print("\n💾 训练结果将保存到: {}/{}_{}_*".format(
        args.output_dir, args.dataset,
        'timestamp'
    ))

    print("\n🚀 开始训练...")
    print("=" * 60)

    # 导入必要的包
    import numpy as np

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建训练器
    trainer = Trainer(args)

    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        print("已保存的checkpoint可以在 {} 找到".format(args.output_dir))
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n📊 后续步骤:")
    print("1. 查看TensorBoard: tensorboard --logdir={}".format(args.output_dir))
    print("2. 评估模型: python evaluate.py --checkpoint outputs/.../checkpoint_best.pth")
    print("3. 如需更好性能，考虑使用云端GPU: 查看 CLOUD_GPU_GUIDE.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
