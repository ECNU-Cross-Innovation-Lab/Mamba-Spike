"""
简单训练示例 - 用于快速测试和学习

这是一个最小化的训练脚本，展示如何使用Mamba-Spike进行训练。
适合初学者理解训练流程，或快速测试代码是否正常工作。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.dataset_loader import prepare_nmnist_dataset
from models.mamba_spike import create_mamba_spike_nmnist


def simple_train():
    """简单的训练函数"""
    print("=" * 60)
    print("Mamba-Spike 简单训练示例")
    print("=" * 60)

    # 1. 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ 使用 CUDA (GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ 使用 MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("✅ 使用 CPU")

    # 2. 准备数据
    print("\n正在加载 N-MNIST 数据集...")
    print("(首次运行会下载数据集，大约需要几分钟)")

    train_loader, test_loader, num_classes = prepare_nmnist_dataset(
        batch_size=16,  # 较小的batch size适合快速测试
        time_window=300000,  # 300ms时间窗口
        dt=1000,  # 1ms时间分辨率
        num_workers=2  # 数据加载线程数
    )

    print(f"✅ 数据加载完成")
    print(f"   训练样本批次: {len(train_loader)}")
    print(f"   测试样本批次: {len(test_loader)}")
    print(f"   类别数: {num_classes}")

    # 3. 创建模型
    print("\n正在创建模型...")
    model = create_mamba_spike_nmnist(num_classes=num_classes)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建完成")
    print(f"   参数量: {num_params:,}")

    # 4. 设置训练参数
    epochs = 3  # 只训练3个epoch用于演示
    learning_rate = 1e-3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n训练设置:")
    print(f"   Epochs: {epochs}")
    print(f"   学习率: {learning_rate}")
    print(f"   损失函数: CrossEntropyLoss")
    print(f"   优化器: Adam")

    # 5. 训练循环
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        # 训练统计
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total

        # 测试阶段
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += criterion(output, target).item()

                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        # 测试统计
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total

        # 打印epoch结果
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  测试 - Loss: {avg_test_loss:.4f}, Acc: {test_acc:.2f}%")
        print("-" * 60)

    # 6. 保存模型
    print("\n保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
    }, 'simple_model.pth')
    print(f"✅ 模型已保存到: simple_model.pth")

    # 7. 总结
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最终测试准确率: {test_acc:.2f}%")
    print(f"\n注意: 这只是演示训练 ({epochs} epochs)")
    print(f"要达到论文中的性能 (~99.5%), 需要训练更多epochs")
    print(f"\n运行完整训练:")
    print(f"  python train.py --dataset nmnist --epochs 100 --batch-size 32")
    print("=" * 60)


if __name__ == "__main__":
    try:
        simple_train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
