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

def create_clients(n_clients, model_template, foogd_template, client_loaders, device, model_type='densenet169', compute_aug_features=True, freeze_bn=True):
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
            device=device,
            compute_aug_features=compute_aug_features,
            freeze_bn=freeze_bn
        )
        clients.append(client)

    return clients

# 在 federated_training 函数中调用 Server 时也要传 foogd_module
# server = FLServer(global_model, foogd_module, device)


def federated_training(args):
    """联邦学习训练主函数 (增强版: 修复检查点与评估逻辑)"""
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

    # -------------------------------------------------
    # 1. 增强的检查点加载逻辑 (断点续训/恢复)
    # -------------------------------------------------
    start_round = 0
    best_acc = 0.0

    # 尝试自动寻找最新的检查点 (防止意外中断)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        if checkpoints and args.resume: # 假设你在 argparse 加了 --resume
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"!!! 发现检查点，准备恢复: {latest_ckpt} !!!")
            ckpt = torch.load(latest_ckpt)

            # 恢复 Server
            server.global_model.load_state_dict(ckpt['global_model_state_dict'])
            if foogd_module and ckpt['foogd_state_dict']:
                foogd_module.load_state_dict(ckpt['foogd_state_dict'])

            # 恢复训练进度
            start_round = ckpt['round']
            training_history = ckpt['training_history']
            best_acc = training_history.get('best_acc', 0.0) # 如果历史里没存，就默认0

            # 注意：Client 的状态将在创建 Client 后恢复
            saved_client_states = ckpt.get('client_states', None)
        else:
            saved_client_states = None
    else:
        saved_client_states = None

    # 创建客户端
    print("\n创建客户端...")
    clients = create_clients(
        args.n_clients, global_model, foogd_module, client_loaders, device, args.model_type,
        compute_aug_features=args.compute_aug_features,
        freeze_bn=args.freeze_bn
    )

    # [关键修复 1] 如果有检查点，恢复每个 Client 的 Head-P
    if saved_client_states is not None:
        print("正在恢复客户端个性化状态 (Head-P)...")
        for i, client in enumerate(clients):
            if i in saved_client_states:
                # 仅加载个性化头，或者整个模型状态
                # 由于 create_clients 已经拷贝了 global weights，这里 load_state_dict 会覆盖 head_p
                client.model.load_state_dict(saved_client_states[i], strict=False)

    # 如果没有恢复历史，初始化历史记录
    if 'training_history' not in locals():
        training_history = {
            'rounds': [],
            'train_losses': [],
            'test_accuracies': [],
            'avg_person_acc': [], # [关键修复 2] 确保有这个 key
            'avg_global_local_acc': [],
            'inc_accuracies': [],
            'test_losses': [],
            'near_auroc': [],
            'far_auroc': [],
            'best_acc': 0.0
        }

    # 联邦学习训练循环
    print(f"\n开始联邦学习训练，从第 {start_round + 1} 轮 到 {args.communication_rounds} 轮...")

    for round_num in range(start_round, args.communication_rounds):
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

            # 客户端本地训练 (传入当前轮次用于动态权重调度)
            client_update, client_loss = client.train_step(
                local_epochs=args.local_epochs,
                current_round=round_num  # 新增参数：当前轮次
            )

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

        # === 评估阶段 ===
        if (round_num + 1) % args.eval_frequency == 0 or round_num == args.communication_rounds - 1:
            print("\n评估模型...")

            # 1. Server Global Model 评估
            # [修复] 取消每5轮一次的限制，改为每轮都评估，确保数据长度一致
            test_metrics = server.evaluate_global_model(
                test_loader, near_ood_loader, far_ood_loader, inc_loader
            )

            # 2. [关键修复 2] 方案 B：严谨的本地测试集评估
            client_acc_p_list = []
            client_acc_g_list = []
            print(f"  正在评估 {len(clients)} 个客户端的个性化性能 (严谨模式: 仅评估本地可见类别)...")

            for client in clients:
                # --- 方案 B 代码段开始 ---
                subset = client.train_loader.dataset
                full_train_dataset = subset.dataset
                client_indices = subset.indices
                seen_classes = set()
                for idx in client_indices:
                    seen_classes.add(full_train_dataset.labels[idx])

                test_dataset = test_loader.dataset
                local_test_indices = [i for i, label in enumerate(test_dataset.labels) if label in seen_classes]

                if len(local_test_indices) == 0:
                    continue

                local_test_subset = torch.utils.data.Subset(test_dataset, local_test_indices)
                local_test_loader = torch.utils.data.DataLoader(
                    local_test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
                )

                c_metrics = client.evaluate(local_test_loader)
                client_acc_p_list.append(c_metrics['acc_p'])
                client_acc_g_list.append(c_metrics['acc_g'])
                # --- 方案 B 代码段结束 ---

            avg_acc_p = sum(client_acc_p_list) / len(client_acc_p_list) if client_acc_p_list else 0.0
            avg_acc_g_local = sum(client_acc_g_list) / len(client_acc_g_list) if client_acc_g_list else 0.0

            # 3. 记录日志 (确保写入 history)
            training_history['rounds'].append(round_num + 1)
            training_history['test_accuracies'].append(test_metrics['id_accuracy'])
            training_history['avg_person_acc'].append(avg_acc_p) # 记录个性化精度
            training_history['avg_global_local_acc'].append(avg_acc_g_local)
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

            print(f"  [Server] Global Head-G ID准确率: {test_metrics['id_accuracy']:.4f}")
            print(f"  [Clients] Average Head-P ID准确率: {avg_acc_p:.4f} (个性化严谨指标)")

            # 4. 保存 Best Model (增加 client states)
            current_acc = avg_acc_p # 建议：pFL 通常以个性化精度作为 Best 的标准，或者用 id_accuracy，你自己定
            if current_acc > best_acc:
                best_acc = current_acc
                training_history['best_acc'] = best_acc # 更新历史中的 best

                # 收集所有 Client 的状态
                client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}

                torch.save({
                    'round': round_num + 1,
                    'global_model_state_dict': server.global_model.state_dict(),
                    'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
                    'client_states': client_states, # [关键] 保存客户端状态
                    'best_acc': best_acc,
                    'config': vars(args)
                }, os.path.join(experiment_dir, "best_model.pth"))
                print(f"  ★ 新最优模型 (Avg Person Acc: {best_acc:.4f}) 已保存")

            # 5. 保存定期 Checkpoint (增加 client states)
            if (round_num + 1) % args.save_frequency == 0:
                client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}
                torch.save({
                    'round': round_num + 1,
                    'global_model_state_dict': server.global_model.state_dict(),
                    'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
                    'client_states': client_states, # [关键] 保存客户端状态
                    'training_history': training_history, # 保存完整历史以便恢复
                    'config': vars(args)
                }, os.path.join(experiment_dir, "checkpoints", f"round_{round_num + 1}.pth"))
                print(f"检查点已保存: round_{round_num + 1}.pth")

    # 训练完成
    print(f"\n联邦学习训练完成!")

    # 保存最终模型 (包含所有客户端状态)
    final_model_path = os.path.join(experiment_dir, "final_model.pth")
    client_states = {i: client.model.state_dict() for i, client in enumerate(clients)}
    torch.save({
        'round': args.communication_rounds,
        'global_model_state_dict': server.global_model.state_dict(),
        'foogd_state_dict': foogd_module.state_dict() if foogd_module else None,
        'client_states': client_states, # [关键] 保存客户端状态
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

        # 恢复客户端状态 (如果存在)
        if 'client_states' in checkpoint:
            print("  正在恢复客户端个性化状态...")
            for i, client in enumerate(clients):
                if i in checkpoint['client_states']:
                    client.model.load_state_dict(checkpoint['client_states'][i], strict=False)

        print(f"  已恢复到第 {checkpoint['round']} 轮的状态 (Best Acc: {checkpoint['best_acc']:.4f})")
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
    parser.add_argument('--local_epochs', type=int, default=4,
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
    parser.add_argument('--resume', action='store_true', default=False,
                       help='是否从最新检查点恢复训练')  # [新增]
    parser.add_argument('--compute_aug_features', action='store_true', default=True,
                       help='是否计算增强数据的特征（显存不足时可设为False）')
    parser.add_argument('--freeze_bn', action='store_true', default=True,
                       help='是否冻结BN统计量（默认True，使用预训练ImageNet统计量）')

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