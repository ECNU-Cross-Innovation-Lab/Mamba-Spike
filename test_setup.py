"""
测试安装和环境设置
运行此脚本来验证所有依赖是否正确安装
"""

import sys

def test_imports():
    """测试所有必需的包是否可以导入"""
    print("=" * 60)
    print("测试Python包导入...")
    print("=" * 60)

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'tonic': 'Tonic (事件数据处理)',
        'snntorch': 'snnTorch (脉冲神经网络)',
        'einops': 'Einops',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'tensorboard': 'TensorBoard'
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name:30s} - 已安装")
        except ImportError as e:
            print(f"❌ {name:30s} - 未安装")
            failed.append(package)

    if failed:
        print("\n" + "=" * 60)
        print(f"❌ {len(failed)} 个包未安装:")
        print("=" * 60)
        for pkg in failed:
            print(f"  - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有依赖包已正确安装!")
    print("=" * 60)
    return True


def test_device():
    """测试可用的计算设备"""
    import torch

    print("\n" + "=" * 60)
    print("检测计算设备...")
    print("=" * 60)

    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")

    # CPU
    print(f"✅ CPU 可用")

    # CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   设备数量: {torch.cuda.device_count()}")
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"ℹ️  CUDA 不可用 (将使用CPU)")

    # MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"✅ MPS (Apple Silicon) 可用")
    else:
        print(f"ℹ️  MPS 不可用")

    print("=" * 60)


def test_model():
    """测试模型创建和前向传播"""
    import torch
    from models.mamba_spike import create_mamba_spike_nmnist

    print("\n" + "=" * 60)
    print("测试模型创建...")
    print("=" * 60)

    try:
        # 创建模型
        model = create_mamba_spike_nmnist()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功")
        print(f"   参数量: {num_params:,}")

        # 测试前向传播
        batch_size = 2
        time_steps = 10
        x = torch.randn(batch_size, time_steps, 2, 34, 34)

        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"✅ 前向传播成功")
        print(f"   输入形状: {tuple(x.shape)}")
        print(f"   输出形状: {tuple(output.shape)}")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """测试数据加载器"""
    print("\n" + "=" * 60)
    print("测试数据加载器...")
    print("=" * 60)
    print("注意: 首次运行会下载数据集，可能需要几分钟")
    print("=" * 60)

    try:
        from data.dataset_loader import NeuromorphicDataset

        # 创建数据集对象（不实际加载数据）
        dataset = NeuromorphicDataset(
            dataset_name='nmnist',
            data_dir='./data',
            time_window=10000,
            dt=1000
        )

        print(f"✅ 数据加载器创建成功")
        print(f"   数据集: N-MNIST")
        print(f"   类别数: {dataset.num_classes}")
        print(f"   传感器尺寸: {dataset.sensor_size}")

        print("\n如需测试完整数据加载 (会下载数据集):")
        print("python -c \"from data.dataset_loader import prepare_nmnist_dataset; prepare_nmnist_dataset(batch_size=4, num_workers=0)\"")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n")
    print("🔍 Mamba-Spike 环境测试")
    print("=" * 60)

    results = []

    # 测试导入
    results.append(("包导入", test_imports()))

    # 测试设备
    test_device()

    # 测试模型
    results.append(("模型创建", test_model()))

    # 测试数据加载
    results.append(("数据加载器", test_data_loader()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 所有测试通过! 你可以开始训练了:")
        print("   python train.py --dataset nmnist --epochs 5 --batch-size 32")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复")
        print("   如需帮助，请查看 QUICK_START.md 或提交 GitHub Issue")

    print("\n")
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
