#!/usr/bin/env python3
"""
测试脚本 - 验证pFL-FOOGD系统的各个组件

作者: Claude Code
日期: 2025-11-22
"""

import torch
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_fedrod_model
from data_utils import create_federated_loaders
from client import FLClient
from server import FLServer
from eval_utils import generate_evaluation_report


def test_models():
    """测试模型组件"""
    print("=== 测试模型组件 ===")

    # 测试模型创建
    model, foogd_module = create_fedrod_model(
        model_type='densenet169',
        num_classes=54,
        use_foogd=True
    )

    print(f"[OK] FedRoD模型创建成功")
    print(f"  - 骨干网络: DenseNet-169")
    print(f"  - 通用头参数: {sum(p.numel() for name, p in model.named_parameters() if 'head_g' in name):,}")
    print(f"  - 个性化头参数: {sum(p.numel() for name, p in model.named_parameters() if 'head_p' in name):,}")

    if foogd_module:
        print(f"[OK] FOOGD模块创建成功")
        print(f"  - FOOGD参数: {sum(p.numel() for p in foogd_module.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    logits_g, logits_p, features = model(dummy_input)

    print(f"[OK] 前向传播测试通过")
    print(f"  - 输入尺寸: {dummy_input.shape}")
    print(f"  - 通用头输出: {logits_g.shape}")
    print(f"  - 个性化头输出: {logits_p.shape}")
    print(f"  - 特征向量: {features.shape}")

    # 测试FOOGD模块
    if foogd_module:
        ksd_loss, sm_loss, ood_scores = foogd_module(features)
        print(f"[OK] FOOGD模块测试通过")
        print(f"  - KSD损失: {ksd_loss.item():.4f}")
        print(f"  - 评分匹配损失: {sm_loss.item():.4f}")
        print(f"  - OOD分数: {ood_scores.shape}")

    print()


def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载 ===")

    data_root = "./Plankton_OOD_Dataset"

    try:
        # 测试联邦学习数据加载器
        client_loaders, test_loader, near_ood_loader, far_ood_loader = create_federated_loaders(
            data_root=data_root,
            n_clients=3,
            alpha=0.1,
            batch_size=4,
            image_size=224
        )

        print(f"[OK] 联邦学习数据加载器创建成功")
        print(f"  - 客户端数量: {len(client_loaders)}")

        # 测试数据批次
        for client_id, loader in enumerate(client_loaders):
            for images, labels in loader:
                print(f"  - 客户端 {client_id}: 批次尺寸 {images.shape}, 标签 {labels.shape}")
                break

        if test_loader:
            print(f"  - 测试集: {len(test_loader.dataset)} 样本")
        if near_ood_loader:
            print(f"  - Near-OOD: {len(near_ood_loader.dataset)} 样本")
        if far_ood_loader:
            print(f"  - Far-OOD: {len(far_ood_loader.dataset)} 样本")

    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        print("  请确保数据集已正确划分并放置在 ./data 目录")

    print()


def test_client_server():
    """测试客户端和服务端"""
    print("=== 测试客户端和服务端 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型和数据
    model, foogd_module = create_fedrod_model()
    model = model.to(device)
    if foogd_module:
        foogd_module = foogd_module.to(device)

    data_root = "./Plankton_OOD_Dataset"
    try:
        client_loaders, test_loader, _, _ = create_federated_loaders(
            data_root, n_clients=2, batch_size=4, image_size=224
        )

        # 测试服务端
        server = FLServer(model, device)
        print(f"[OK] 服务端创建成功")

        # 测试客户端
        client = FLClient(
            client_id=0,
            model=model,
            foogd_module=foogd_module,
            train_loader=client_loaders[0],
            device=device,
            compute_aug_features=True,  # 测试时默认计算增强特征
            freeze_bn=True  # 测试时默认冻结BN统计量
        )
        print(f"[OK] 客户端创建成功")

        # 测试客户端训练
        generic_params, train_loss = client.train_step(local_epochs=1)
        print(f"[OK] 客户端训练测试通过")
        print(f"  - 训练损失: {train_loss:.4f}")
        print(f"  - 通用参数数量: {len(generic_params)}")

        # 测试服务端聚合
        client_updates = [generic_params, generic_params]
        client_sample_sizes = [100, 150]

        aggregated_params = server.aggregate(client_updates, client_sample_sizes)
        print(f"[OK] 服务端聚合测试通过")
        print(f"  - 聚合参数数量: {len(aggregated_params)}")

        # 测试模型评估
        accuracy, eval_loss = client.evaluate(test_loader)
        print(f"[OK] 模型评估测试通过")
        print(f"  - 评估准确率: {accuracy:.4f}")
        print(f"  - 评估损失: {eval_loss:.4f}")

    except Exception as e:
        print(f"[ERROR] 客户端/服务端测试失败: {e}")

    print()


def test_evaluation():
    """测试评估工具"""
    print("=== 测试评估工具 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model, foogd_module = create_fedrod_model()
    model = model.to(device)
    if foogd_module:
        foogd_module = foogd_module.to(device)

    data_root = "./Plankton_OOD_Dataset"
    try:
        _, test_loader, near_ood_loader, far_ood_loader = create_federated_loaders(
            data_root, n_clients=2, batch_size=4, image_size=224
        )

        # 测试评估报告生成
        output_dir = "./test_evaluation"
        report = generate_evaluation_report(
            model, foogd_module, test_loader, near_ood_loader,
            far_ood_loader, device, output_dir
        )

        print(f"[OK] 评估报告生成成功")
        if 'id_classification' in report:
            acc = report['id_classification']['accuracy']
            print(f"  - ID分类准确率: {acc:.4f}")

        if 'near_ood_detection' in report:
            auroc = report['near_ood_detection']['auroc']
            print(f"  - Near-OOD AUROC: {auroc:.4f}")

        if 'far_ood_detection' in report:
            auroc = report['far_ood_detection']['auroc']
            print(f"  - Far-OOD AUROC: {auroc:.4f}")

        print(f"  - 评估结果保存在: {output_dir}")

    except Exception as e:
        print(f"[ERROR] 评估工具测试失败: {e}")

    print()


def test_memory_usage():
    """测试内存使用情况"""
    print("=== 测试内存使用情况 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试不同批次的GPU内存使用
    batch_sizes = [16, 32, 64]

    for batch_size in batch_sizes:
        try:
            model, _ = create_fedrod_model()
            model = model.to(device)

            # 模拟前向传播
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
            _ = model(dummy_input)

            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                print(f"[OK] 批次大小 {batch_size}: GPU内存使用 {memory_allocated:.2f} GB")
            else:
                print(f"[OK] 批次大小 {batch_size}: CPU模式")

            # 清理内存
            del model, dummy_input
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] 批次大小 {batch_size} 测试失败: {e}")

    print()


def main():
    """主测试函数"""
    print("pFL-FOOGD系统组件测试")
    print("=" * 50)

    # 运行所有测试
    test_models()
    test_data_loading()
    test_client_server()
    test_evaluation()
    test_memory_usage()

    print("=" * 50)
    print("测试完成!")
    print("\n下一步:")
    print("1. 确保数据集已正确划分并放置在 ./data 目录")
    print("2. 运行: python train_federated.py 开始联邦学习训练")
    print("3. 查看实验结果保存在 ./experiments 目录")


if __name__ == "__main__":
    main()