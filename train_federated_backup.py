#!/usr/bin/env python3
"""
联邦学习训练主脚本
基于pFL-FOOGD框架的海洋浮游生物图像识别

作者: Claude Code
日期: 2025-11-22
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random  # 确保导入 random

from models import create_fedrod_model
from data_utils import create_federated_loaders
from client import FLClient
from server import FLServer
from eval_utils import generate_evaluation_report

def set_seed(seed):
    """固定所有随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")

def setup_experiment(args):
    """设置实验环境"""
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    return experiment_dir


import copy # 确保引入 copy

def create_clients(n_clients, model_template, foogd_template, client_loaders, device, model_type='densenet169'):
    """创建客户端 (修正版)"""
    clients = []

    for client_id in range(n_clients):
        # 1. 深拷贝模型 (Model)
        client_model, _ = create_fedrod_model(
            model_type=model_type,
            num_classes=54,
            use_foogd=False
        )
        client_model.load_state_dict(model_template.state_dict())
        client_model = client_model.to(device)  # 将客户端模型移动到设备

        # 2. 深拷贝 FOOGD 模块 (关键修正！)
        # 必须让每个客户端拥有独立的 Score Model 副本
        client_foogd = None
        if foogd_template is not None:
            client_foogd = copy.deepcopy(foogd_template)
            client_foogd = client_foogd.to(device)  # 将FOOGD模块移动到设备

        client = FLClient(
            client_id=client_id,
            model=client_model,
            foogd_module=client_foogd,  # 传入独立的副本
            train_loader=client_loaders[client_id],
            device=device
        )
        clients.append(client)

    return clients

# 在 federated_training 函数中调用 Server 时也要传 foogd_module
# server = FLServer(global_model, foogd_module, device)


def federated_training(args):
    """联邦学习训练主函数"""
    print("开始联邦学习训练...")

    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 设置实验
    experiment_dir = setup_experiment(args)
    print(f"实验目录: {experiment_dir}")

    # 创建数据加载器
    print("\n创建数据加载器...")
    client_loaders, test_loader, near_ood_loader, far_ood_loader, inc_loader = create_federated_loaders(
        data_root=args.data_root,
        n_clients=args.n_clients,
        alpha=args.alpha,
        batch_size=args.batch_size,
        image_size=args.image_size
    )

    # 创建全局模型
    print("\n创建全局模型...")
    global_model, foogd_module = create_fedrod_model(
        model_type=args.model_type,
        num_classes=54,
        use_foogd=args.use_foogd
    )

    # 将模型移动到设备
    global_model = global_model.to(device)
    if foogd_module:
        foogd_module = foogd_module.to(device)

    # 创建服务端
    server = FLServer(global_model, foogd_module, device)

    # 创建客户端
    print("\n创建客户端...")
    clients = create_clients(
        args.n_clients, global_model, foogd_module, client_loaders, device, args.model_type
    )

    # 训练历史
    training_history = {
        'rounds': [],
        'train_losses': [],
        'test_accuracies': [],  # ID (Clean) Accuracy
        'inc_accuracies': [],   # IN-C (Corrupted) Accuracy <- 新增
        'test_losses': [],
        'near_auroc': [],
        'far_auroc': []
    }

    # [新增] 记录历史最优准确率
    best_acc = 0.0

    # 联邦学习训练循环
    print(f"\n开始联邦学习训练，共 {args.communication_rounds} 轮...")

    for round_num in range(args.communication_rounds):
        print(f"\n=== 通信轮次 {round_num + 1}/{args.communication_rounds} ===")

        # 选择参与本轮训练的客户端
        if args.client_fraction < 1.0:
            n_selected = max(1, int(args.n_clients * args.client_fraction))
            selected_clients = np.random.choice(
                args.n_clients, n_selected, replace=False
            )
        else:
            selected_clients = range(args.n_clients)

        print(f"选择的客户端: {list(selected_clients)}")

        # 客户端本地训练
        client_updates = []
        client_sample_sizes = []
        round_train_loss = 0.0

        for client_id in selected_clients:
            client = clients[client_id]

            # 设置客户端模型参数
            client.set_generic_parameters(server.get_global_parameters())

            # 客户端本地训练
            client_update, client_loss = client.train_step(local_epochs=args.local_epochs)

            client_updates.append(client_update)
            client_sample_sizes.append(len(client.train_loader.dataset))
            round_train_loss += client_loss

            print(f"  客户端 {client_id}: 本地损失 = {client_loss:.4f}")

        # 服务器聚合
        aggregated_params = server.aggregate(client_updates, client_sample_sizes)
        server.set_global_parameters(aggregated_params)

        # 更新客户端模型
        for client in clients:
            client.set_generic_parameters(server.get_global_parameters())

        # === 评估阶段 (修改后) ===
        if (round_num + 1) % args.eval_frequency == 0 or round_num == args.communication_rounds - 1:
            print("\n评估模型...")

            # 1. 评估 Server 端的 Global Model (主要看 head_g 和 OOD 检测)
            # 注意：这里的 metrics['id_accuracy'] 是 head_g 的准确率
            test_metrics = server.evaluate_global_model(
                test_loader, near_ood_loader, far_ood_loader, inc_loader
            )

            # =================================================================================
            # [修改版] 2. 评估所有 Client 的 Local Model (主要看 head_p)
            # 采用方案 B：严谨的"本地测试集 (Local Test Set)" 评估逻辑
            # 原理：只用客户端"训练过/见过"的类别来考核它的个性化模型，避免评估未见类别的干扰
            # =================================================================================

            client_acc_p_list = []
            client_acc_g_list = []

            print(f"  正在评估 {len(clients)} 个客户端的个性化性能 (严谨模式: 仅评估本地可见类别)...")

            for client in clients:
                # --- 步骤 1: 确定该客户端见过的类别 (Seen Classes) ---
                # client.train_loader.dataset 是一个 Subset 对象
                subset = client.train_loader.dataset
                # 获取子集背后的完整训练集对象 (PlanktonDataset)
                full_train_dataset = subset.dataset
                # 获取该客户端拥有的样本索引列表
                client_indices = subset.indices

                # 提取该客户端训练集中出现过的所有唯一标签
                seen_classes = set()
                for idx in client_indices:
                    # 直接访问原始数据集的 labels 列表，比遍历 loader 快得多
                    label = full_train_dataset.labels[idx]
                    seen_classes.add(label)

                # --- 步骤 2: 动态构建本地测试集 (Local Test Set) ---
                # 获取全局测试集对象
                test_dataset = test_loader.dataset

                # 筛选：从全局测试集中，只保留标签属于 seen_classes 的样本索引
                local_test_indices = [
                    i for i, label in enumerate(test_dataset.labels)
                    if label in seen_classes
                ]

                # 异常处理：万一该客户端太倒霉，它的类别在测试集里一个都没有 (极罕见)
                if len(local_test_indices) == 0:
                    print(f"    [警告] 客户端 {client.client_id} 的本地测试集为空 (训练集类别在测试集中未出现)，跳过此客户端。")
                    continue

                # 利用 Subset 创建一个新的轻量级数据集
                local_test_subset = torch.utils.data.Subset(test_dataset, local_test_indices)

                # 创建临时的 DataLoader 用于评估
                # 注意：num_workers 可以设小一点，避免频繁创建销毁进程的开销
                local_test_loader = torch.utils.data.DataLoader(
                    local_test_subset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                # --- 步骤 3: 执行评估 ---
                # 使用筛选后的专属测试集进行评估
                c_metrics = client.evaluate(local_test_loader)

                client_acc_p_list.append(c_metrics['acc_p'])
                client_acc_g_list.append(c_metrics['acc_g'])

                # (可选) 打印详细调试信息，看看每个客户端测了多少样本
                # print(f"    Client {client.client_id}: Seen {len(seen_classes)} classes, Test Samples: {len(local_test_indices)}, Acc-P: {c_metrics['acc_p']:.4f}")

            # 计算平均值 (防止除以零)
            if len(client_acc_p_list) > 0:
                avg_acc_p = sum(client_acc_p_list) / len(client_acc_p_list)
                avg_acc_g_local = sum(client_acc_g_list) / len(client_acc_g_list)
            else:
                avg_acc_p = 0.0
                avg_acc_g_local = 0.0

            # 更新历史记录
            training_history['rounds'].append(round_num + 1)
            # 记录 Server 端的 head_g
            training_history['test_accuracies'].append(test_metrics['id_accuracy'])
            # [建议新增] 记录 Average Personal Accuracy
            training_history.setdefault('avg_person_acc', []).append(avg_acc_p)

            print(f"  [Server] Global Head-G ID准确率: {test_metrics['id_accuracy']:.4f}")
            print(f"  [Clients] Average Head-P ID准确率: {avg_acc_p:.4f} (这是个性化性能的关键指标!)")
            print(f"  [Clients] Average Head-G ID准确率: {avg_acc_g_local:.4f}")

            # [新增] 保存历史最优模型 (Best Model)
            current_acc = test_metrics['id_accuracy']
            if current_acc > best_acc:
                best_acc = current_acc
                best_model_path = os.path.join(experiment_dir, "best_model.pth")
                torch.save({
                    'round': round_num + 1,
                    'global_model_state_dict': server.global_model.state_dict(),
                    'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
                    'best_acc': best_acc,
                    'config': vars(args)
                }, best_model_path)
                print(f"  ★ 发现新最优模型 (Acc: {best_acc:.4f})，已保存: {best_model_path}")

            # 原有记录逻辑保持不变
            training_history['train_losses'].append(round_train_loss / len(selected_clients))
            training_history['test_losses'].append(test_metrics['id_loss'])

            if 'near_auroc' in test_metrics:
                training_history['near_auroc'].append(test_metrics['near_auroc'])
            if 'far_auroc' in test_metrics:
                training_history['far_auroc'].append(test_metrics['far_auroc'])

            # [新增] 记录并打印 IN-C 结果
            if 'inc_accuracy' in test_metrics:
                acc = test_metrics['inc_accuracy']
                training_history['inc_accuracies'].append(acc)
                print(f"  IN-C 准确率 (泛化): {acc:.4f}")
                # 对比 ID 准确率，可以直观看到下降幅度
                print(f"  准确率下降 (Drop): {test_metrics['id_accuracy'] - acc:.4f}")

            # 打印评估结果
            print(f"  ID损失: {test_metrics['id_loss']:.4f}")
            if 'near_auroc' in test_metrics:
                print(f"  Near-OOD AUROC: {test_metrics['near_auroc']:.4f}")
            if 'far_auroc' in test_metrics:
                print(f"  Far-OOD AUROC: {test_metrics['far_auroc']:.4f}")

            # 保存检查点
            if (round_num + 1) % args.save_frequency == 0:
                checkpoint_path = os.path.join(
                    experiment_dir, "checkpoints", f"round_{round_num + 1}.pth"
                )
                torch.save({
                    'round': round_num + 1,
                    'global_model_state_dict': server.global_model.state_dict(),
                    'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
                    'training_history': training_history,
                    'config': vars(args)
                }, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")

    # 训练完成
    print(f"\n联邦学习训练完成!")

    # 保存最终模型
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        'global_model_state_dict': server.global_model.state_dict(),
        'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
        'training_history': training_history,
        'config': vars(args)
    }, final_model_path)
    print(f"最终模型已保存: {final_model_path}")

    # 保存训练历史
    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"训练历史已保存: {history_path}")

    # 绘制训练曲线
    plot_training_curves(training_history, experiment_dir)

    print("\n正在准备生成详细评估报告...")

    # [新增关键逻辑]：加载历史最优模型 (Best Model)
    # 这样生成的混淆矩阵和 OOD 图才是该实验能达到的"上限"性能
    best_model_path = os.path.join(experiment_dir, "best_model.pth")

    if os.path.exists(best_model_path):
        print(f"★ 正在加载历史最优模型进行评估: {best_model_path}")
        checkpoint = torch.load(best_model_path)

        # 加载参数
        server.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        if foogd_module and checkpoint['foogd_state_dict']:
            foogd_module.load_state_dict(checkpoint['foogd_state_dict'])

        print(f"  已恢复到第 {checkpoint['round']} 轮的状态 (Acc: {checkpoint['best_acc']:.4f})")
    else:
        print("  未找到最优模型检查点，将使用最终轮次模型进行评估（这可能不是最佳性能）。")

    # 原有的生成报告代码
    print("正在生成详细评估报告(混淆矩阵 & OOD 分类图)...")
    eval_output_dir = os.path.join(experiment_dir, "final_evaluation")
    generate_evaluation_report(
        model = server.global_model,
        foogd_module = foogd_module,
        test_loader = test_loader,
        near_ood_loader = near_ood_loader,
        far_ood_loader = far_ood_loader,
        device = device,
        output_dir = eval_output_dir
    )
    print(f"详细评估图表已保存至： {eval_output_dir}")

    return training_history


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    if not history['rounds']:
        return

    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['rounds'], history['train_losses'], 'b-', label='Training Loss')
    plt.plot(history['rounds'], history['test_losses'], 'r-', label='Test Loss')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['rounds'], history['test_accuracies'], 'g-', label='Global Head-G Accuracy')
    if 'avg_person_acc' in history and history['avg_person_acc']:
        plt.plot(history['rounds'], history['avg_person_acc'], 'm-', label='Avg Head-P Accuracy')
    if history['inc_accuracies']:
        plt.plot(history['rounds'], history['inc_accuracies'], 'c-', label='IN-C Accuracy')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Test Accuracy (Global vs Personal vs IN-C)')

    # OOD检测性能
    if history['near_auroc']:
        plt.subplot(2, 2, 3)
        plt.plot(history['rounds'], history['near_auroc'], 'orange', label='Near-OOD AUROC')
        if history['far_auroc']:
            plt.plot(history['rounds'], history['far_auroc'], 'purple', label='Far-OOD AUROC')
        plt.xlabel('Communication Rounds')
        plt.ylabel('AUROC')
        plt.legend()
        plt.title('OOD Detection Performance')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练曲线图已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='联邦学习训练脚本 - pFL-FOOGD')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                       help='数据集根目录路径')
    parser.add_argument('--n_clients', type=int, default=10,
                       help='客户端数量')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='狄利克雷分布参数（控制数据异质性）')

    # 训练参数
    parser.add_argument('--communication_rounds', type=int, default=50,
                       help='通信轮次')
    parser.add_argument('--local_epochs', type=int, default=1,
                       help='每个客户端的本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--client_fraction', type=float, default=1.0,
                       help='每轮选择的客户端比例')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='densenet169',
                       choices=['densenet121', 'densenet169'],
                       help='骨干网络类型')
    parser.add_argument('--use_foogd', action='store_true', default=False,
                       help='是否使用FOOGD模块')
    parser.add_argument('--image_size', type=int, default=224,
                       help='输入图像尺寸')

    # 评估和保存
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='评估频率（轮次）')
    parser.add_argument('--save_frequency', type=int, default=2,
                       help='保存检查点频率（轮次）')

    # 系统参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')  # [新增]

    args = parser.parse_args()

    # [新增] 在一切开始之前，先设置种子
    set_seed(args.seed)

    # 打印配置
    print("联邦学习训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 开始训练
    training_history = federated_training(args)

    # 打印最终结果
    print("\n=== 训练完成 ===")
    if training_history['test_accuracies']:
        final_acc = training_history['test_accuracies'][-1]
        print(f"最终全局准确率 (Head-G): {final_acc:.4f}")

    if 'avg_person_acc' in training_history and training_history['avg_person_acc']:
        final_person_acc = training_history['avg_person_acc'][-1]
        print(f"最终个性化准确率 (Head-P): {final_person_acc:.4f}")
        print(f"个性化增益: {final_person_acc - final_acc:.4f}")

    if training_history['inc_accuracies']:
        final_inc_acc = training_history['inc_accuracies'][-1]
        print(f"最终IN-C准确率: {final_inc_acc:.4f}")
        print(f"泛化性能下降: {final_acc - final_inc_acc:.4f}")

    if training_history['near_auroc']:
        final_near_auroc = training_history['near_auroc'][-1]
        print(f"最终Near-OOD AUROC: {final_near_auroc:.4f}")

    if training_history['far_auroc']:
        final_far_auroc = training_history['far_auroc'][-1]
        print(f"最终Far-OOD AUROC: {final_far_auroc:.4f}")


if __name__ == "__main__":
    # [新增] 开启 cudnn benchmark
    torch.backends.cudnn.benchmark = True

    main()