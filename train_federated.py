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
from data_utils import create_federated_loaders, build_taxonomy_matrix
from client import FLClient
from server import FLServer
from eval_utils import generate_evaluation_report, get_tail_classes, evaluate_accuracy_metrics

def set_seed(seed):
    """固定所有随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed set to: {seed}")


def get_global_tail_classes(client_loaders, threshold_ratio=0.5):
    """
    通过聚合所有客户端的数据来统计全局长尾分布

    Args:
        client_loaders: 客户端DataLoader列表
        threshold_ratio: 定义尾部的比例 (例如 0.5 表示样本数最少的 50% 类别)

    Returns:
        tail_classes_set: 包含尾部类别索引的集合
    """
    print("正在计算全局类别分布以定义尾部类别...")

    # 检查空列表
    if not client_loaders:
        print("  警告: 客户端列表为空，返回空集合")
        return set()

    # 1. 初始化计数器
    global_class_counts = {}

    # 2. 遍历所有客户端
    for client_id, loader in enumerate(client_loaders):
        dataset = loader.dataset

        # 处理 Subset 对象
        if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
            # 这是一个 Subset 对象
            full_dataset = dataset.dataset
            indices = dataset.indices

            # 统计该客户端拥有的样本标签
            for idx in indices:
                label = full_dataset.labels[idx]
                global_class_counts[label] = global_class_counts.get(label, 0) + 1
        else:
            # 直接访问数据集
            # 假设数据集有 labels 属性
            if hasattr(dataset, 'labels'):
                labels = dataset.labels
                for label in labels:
                    global_class_counts[label] = global_class_counts.get(label, 0) + 1
            else:
                # 遍历数据集获取标签
                for _, label in loader:
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    global_class_counts[label] = global_class_counts.get(label, 0) + 1

    # 检查是否有数据
    if not global_class_counts:
        print("  警告: 未找到任何类别数据，返回空集合")
        return set()

    # 3. 排序：按样本数从少到多
    # items() 返回 (label, count) 元组
    sorted_classes = sorted(global_class_counts.items(), key=lambda x: x[1])

    # 打印一下最少和最多的类，确认统计无误
    print(f"  - 全局样本最少的类: ID {sorted_classes[0][0]} (样本数: {sorted_classes[0][1]})")
    print(f"  - 全局样本最多的类: ID {sorted_classes[-1][0]} (样本数: {sorted_classes[-1][1]})")

    # 4. 截取尾部类别 (前 threshold_ratio 比例)
    n_tail = int(len(sorted_classes) * threshold_ratio)
    tail_classes = [c[0] for c in sorted_classes[:n_tail]]

    print(f"  - 定义了 {len(tail_classes)} 个尾部类别 (Tail Classes)")

    return set(tail_classes)


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

def create_clients(n_clients, model_template, foogd_template, client_loaders, device, model_type='densenet169', compute_aug_features=True, freeze_bn=True, base_lr=0.001, algorithm='fedrod', use_taxonomy=False):
    """创建客户端 (修正版)"""
    clients = []
    # 注意：compute_aug_features 和 freeze_bn 参数目前未被使用
    # 但为了保持与调用代码的兼容性而保留

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
            freeze_bn=freeze_bn,
            base_lr=base_lr,  # 传递基础学习率
            algorithm=algorithm,  # 传递算法选择
            use_taxonomy=use_taxonomy  # 传递层级损失开关
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

    # 预计算测试集类别索引映射（优化个性化评估性能）
    print("预计算测试集类别索引映射...")
    test_dataset = test_loader.dataset
    class_to_test_indices = {}
    total_test_samples = 0
    for i, label in enumerate(test_dataset.labels):
        if label >= 0:  # 只处理有效标签（ID类别）
            if label not in class_to_test_indices:
                class_to_test_indices[label] = []
            class_to_test_indices[label].append(i)
            total_test_samples += 1
    print(f"  预计算完成: {len(class_to_test_indices)} 个类别, {total_test_samples} 个样本")

    # 构建分类学矩阵和识别尾部类别
    print("\n构建分类学矩阵和识别尾部类别...")
    try:
        taxonomy_matrix = build_taxonomy_matrix(num_classes=54, device=device)
        print(f"  分类学矩阵构建完成: {taxonomy_matrix.shape}")
    except Exception as e:
        print(f"  警告: 无法构建分类学矩阵: {e}")
        print("  将使用单位矩阵作为默认分类学矩阵")
        taxonomy_matrix = torch.eye(54).to(device)  # 54个ID类别

    # 使用全局数据识别尾部类别
    if client_loaders and len(client_loaders) > 0:
        tail_classes_set = get_global_tail_classes(client_loaders, threshold_ratio=0.5)
        print(f"  全局尾部类别识别完成: {len(tail_classes_set)} 个尾部类别")
    else:
        tail_classes_set = set()
        print("  警告: 无法识别尾部类别，使用空集合")

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
        freeze_bn=args.freeze_bn,
        base_lr=args.base_lr if hasattr(args, 'base_lr') else 0.001,  # 使用参数或默认值
        algorithm=args.algorithm,  # 传递算法选择
        use_taxonomy=args.use_taxonomy  # 传递层级损失开关
    )

    # [关键修复 1] 如果有检查点，恢复每个 Client 的 Head-P
    if saved_client_states is not None:
        print("正在恢复客户端个性化状态 (Head-P)...")
        for i, client in enumerate(clients):
            if i in saved_client_states:
                # 仅加载个性化头，或者整个模型状态
                # 由于 create_clients 已经拷贝了 global weights，这里 load_state_dict 会覆盖 head_p
                client.model.load_state_dict(saved_client_states[i], strict=False)

    # [优化] 预先生成所有客户端的本地测试加载器，避免在训练循环中重复创建进程
    print("正在预生成客户端本地测试集 (加速评估)...")
    client_test_loaders = []
    test_dataset = test_loader.dataset  # 获取完整的测试集

    for client in clients:
        # 获取该客户端在训练集中见过的类别
        subset = client.train_loader.dataset
        full_train_dataset = subset.dataset
        client_indices = subset.indices
        seen_classes = set()
        for idx in client_indices:
            seen_classes.add(full_train_dataset.labels[idx])

        # 筛选测试集中对应的类别
        local_test_indices = [i for i, label in enumerate(test_dataset.labels) if label in seen_classes]

        if len(local_test_indices) > 0:
            local_test_subset = torch.utils.data.Subset(test_dataset, local_test_indices)
            # 创建加载器 (持久化)
            loader = torch.utils.data.DataLoader(
                local_test_subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,  # 既然只创建一次，可以稍微给大点
                pin_memory=True,
                persistent_workers=True  # [新增] 关键！保持子进程存活，避免每轮重复初始化
            )
            client_test_loaders.append(loader)
        else:
            client_test_loaders.append(None)

    print(f"  预生成完成: {len([l for l in client_test_loaders if l is not None])} 个客户端有本地测试集")

    # 如果没有恢复历史，初始化历史记录
    if 'training_history' not in locals():
        training_history = {
            'rounds': [],
            'train_losses': [],
            'test_accuracies': [],      # ID 准确率 (对于 FedRoD 是 Head-P, 对于 FedAvg 是 Global)
            'test_losses': [],
            'tail_accuracies': [],      # 尾部类别准确率
            'hierarchical_errors': [],  # 层级错误
            'hierarchical_ratios': [],  # 层级准确率 (0~1) <--- 新增
            'inc_accuracies': [],       # IN-C 准确率
            'near_auroc': [],           # Near-OOD AUROC
            'far_auroc': [],            # Far-OOD AUROC
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

        # 更新客户端模型 (为下一轮做准备)
        for client in clients:
            client.set_generic_parameters(server.get_global_parameters())

        # ===========================================================
        # 【建议修改】: 评估应该在训练之后进行 (或者使用单独的验证集微调)
        # 此时 Head_P 已经适应了当前的 Backbone，能反映真实的个性化能力
        # ===========================================================

        # === 评估阶段 ===
        if (round_num + 1) % args.eval_frequency == 0 or round_num == args.communication_rounds - 1:
            print("\n>>> 开始评估模型指标 <<<")

            # 初始化记录字典
            current_metrics = {}

            # =========================================================
            # 场景 A: FedAvg (所有指标都基于 Global Model)
            # =========================================================
            if args.algorithm == 'fedavg':
                print("评估模式: FedAvg (Global Model 承担所有任务)")

                # 1. 基础指标 (ID Acc, Tail Acc, Hierarchical Error, Hier Accuracy)
                # [修正] 传入 use_head_g=True
                id_acc, tail_acc, hier_err, hier_ratio = evaluate_accuracy_metrics(
                    server.global_model, test_loader, taxonomy_matrix, tail_classes_set, device,
                    use_head_g=True  # <--- 关键修正！告诉它读 Head-G
                )

                # 2. 泛化指标 (IN-C Acc)
                # 注意：这里直接复用 accuracy metrics 函数，但传入 IN-C loader
                # [修正] 传入 use_head_g=True
                inc_acc, _, _, _ = evaluate_accuracy_metrics(
                    server.global_model, inc_loader, taxonomy_matrix, tail_classes_set, device,
                    use_head_g=True  # <--- 关键修正！
                )

                # 3. OOD 指标 (Near/Far AUROC)
                # FedAvg 的 OOD 检测基于 Global Model 的特征和 Score Model
                # 使用 server.evaluate_global_model 获取 OOD 指标
                test_metrics = server.evaluate_global_model(
                    test_loader, near_ood_loader, far_ood_loader, inc_loader
                )

                # 记录
                current_metrics.update({
                    'acc_id': id_acc,
                    'acc_tail': tail_acc,
                    'err_hier': hier_err,
                    'hier_ratio': hier_ratio,  # <--- 新增层级准确率
                    'acc_inc': inc_acc,
                    'near_auroc': test_metrics.get('near_auroc', 0.0),
                    'far_auroc': test_metrics.get('far_auroc', 0.0),
                    'id_accuracy': test_metrics.get('id_accuracy', 0.0),
                    'id_loss': test_metrics.get('id_loss', 0.0)
                })

            # =========================================================
            # 场景 B: FedRoD (分离评估)
            # 1. 准确率类指标 -> 评估 Personalized Model (Clients Avg)
            # 2. OOD类指标    -> 评估 Generic Model (Server Global)
            # =========================================================
            elif args.algorithm == 'fedrod':
                print("评估模式: FedRoD (Acc->Head-P, OOD->Head-G)")

                # --- Part 1: 评估 Head-P (Acc, Tail, Hier, IN-C) ---
                # 策略：在所有 Client 上测试，取平均值
                p_acc_list, p_tail_list, p_hier_list, p_hier_ratio_list, p_inc_list = [], [], [], [], []

                for client in clients:
                    # 使用客户端本地的测试集 (或者全局测试集，取决于你的设定，通常 Acc 看本地或全局皆可)
                    # 建议：为了公平对比，这里统一用 全局测试集 test_loader
                    # 注意：Client 模型 forward 时，FedRoD 会返回 (logits_g, logits_p, z)，
                    # evaluate_accuracy_metrics 内部会根据 use_head_g 参数选择头

                    # FedRoD 的 Acc 依然看 Head-P，所以这里 use_head_g=False (默认值)
                    c_id, c_tail, c_hier, c_hier_ratio = evaluate_accuracy_metrics(
                        client.model, test_loader, taxonomy_matrix, tail_classes_set, device,
                        use_head_g=False # <--- FedRoD 保持默认，读 Head-P
                    )
                    c_inc, _, _, _ = evaluate_accuracy_metrics(
                        client.model, inc_loader, taxonomy_matrix, tail_classes_set, device,
                        use_head_g=False # <--- FedRoD 保持默认
                    )

                    p_acc_list.append(c_id)
                    p_tail_list.append(c_tail)
                    p_hier_list.append(c_hier)
                    p_hier_ratio_list.append(c_hier_ratio)
                    p_inc_list.append(c_inc)

                # 计算平均值
                avg_p_acc = sum(p_acc_list) / len(p_acc_list) if p_acc_list else 0.0
                avg_p_tail = sum(p_tail_list) / len(p_tail_list) if p_tail_list else 0.0
                avg_p_hier = sum(p_hier_list) / len(p_hier_list) if p_hier_list else 0.0
                avg_p_hier_ratio = sum(p_hier_ratio_list) / len(p_hier_ratio_list) if p_hier_ratio_list else 0.0
                avg_p_inc = sum(p_inc_list) / len(p_inc_list) if p_inc_list else 0.0

                # --- Part 2: 评估 Head-G (OOD AUROC) ---
                # 使用 Server 端的 Global Model (只包含 Head-G 和 Backbone)
                # 注意：FedRoD 的 Global Model 在 forward 时通常只过 Head-G
                test_metrics = server.evaluate_global_model(
                    test_loader, near_ood_loader, far_ood_loader, inc_loader
                )

                # 记录
                current_metrics.update({
                    'acc_id': avg_p_acc,       # Head-P
                    'acc_tail': avg_p_tail,    # Head-P
                    'err_hier': avg_p_hier,    # Head-P
                    'hier_ratio': avg_p_hier_ratio,  # <--- 新增层级准确率
                    'acc_inc': avg_p_inc,      # Head-P
                    'near_auroc': test_metrics.get('near_auroc', 0.0), # Head-G
                    'far_auroc': test_metrics.get('far_auroc', 0.0),    # Head-G
                    'id_accuracy': test_metrics.get('id_accuracy', 0.0), # Head-G 的 ID 准确率
                    'id_loss': test_metrics.get('id_loss', 0.0)
                })

            # --- 打印和保存日志 ---
            print(f"  [Result] ID Acc: {current_metrics['acc_id']:.4f}")
            print(f"  [Result] Tail Acc: {current_metrics['acc_tail']:.4f}")
            print(f"  [Result] Hier Error: {current_metrics['err_hier']:.4f} (Lower is better)")
            print(f"  [Result] Hier Score: {current_metrics['hier_ratio']:.4f} (Ratio: 0~1)")
            print(f"  [Result] IN-C Acc: {current_metrics['acc_inc']:.4f}")
            print(f"  [Result] Near AUROC: {current_metrics['near_auroc']:.4f}")
            print(f"  [Result] Far AUROC: {current_metrics['far_auroc']:.4f}")

            # 3. 记录日志 (确保写入 history)
            training_history['rounds'].append(round_num + 1)
            training_history['test_accuracies'].append(current_metrics['acc_id'])
            training_history['test_losses'].append(current_metrics['id_loss'])
            training_history['tail_accuracies'].append(current_metrics['acc_tail'])
            training_history['hierarchical_errors'].append(current_metrics['err_hier'])
            training_history['hierarchical_ratios'].append(current_metrics['hier_ratio'])  # <--- 新增层级准确率
            training_history['inc_accuracies'].append(current_metrics['acc_inc'])
            training_history['near_auroc'].append(current_metrics['near_auroc'])
            training_history['far_auroc'].append(current_metrics['far_auroc'])
            training_history['train_losses'].append(round_train_loss / len(selected_clients))

            # 4. 保存 Best Model (增加 client states)
            # 使用个性化精度作为 Best 的标准（对于 FedRoD）或 ID 准确率（对于 FedAvg）
            current_acc = current_metrics['acc_id']
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
                print(f"  ★ 新最优模型 (Acc: {best_acc:.4f}) 已保存")

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
    parser.add_argument('--algorithm', type=str, default='fedrod',
                       choices=['fedavg', 'fedrod'],
                       help='选择算法: fedavg (纯全局模型) 或 fedrod (双头个性化模型)')
    parser.add_argument('--use_taxonomy', action='store_true', default=False,
                       help='启用层级感知损失函数 (Taxonomy-Aware Loss)')

    # 评估和保存
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='评估频率（轮次）--eval_frequency 1：务必保持。这样你会得到 100 个数据点，曲线会非常完整、细腻。')
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='保存检查点频率（轮次）--save_frequency 10：建议设置。每 10 轮存一个档够用了，防止意外断电能恢复就行，同时节省硬盘空间。')

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
    parser.add_argument('--base_lr', type=float, default=0.001,
                       help='基础学习率（默认0.001，batch_size=64时可尝试0.01）')

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

    # 确保 training_history 存在且包含必要数据
    if not training_history or 'test_accuracies' not in training_history:
        print("警告: 训练历史数据不完整")
        return

    # 全局准确率
    if training_history['test_accuracies']:
        final_acc = training_history['test_accuracies'][-1]
        print(f"最终全局准确率 (Head-G): {final_acc:.4f}")
    else:
        print("警告: 未找到全局准确率数据")
        final_acc = 0.0

    # 个性化准确率 (对于 FedRoD 是 Head-P, 对于 FedAvg 与全局相同)
    # 注意: 现在 test_accuracies 对于 FedRoD 已经是 Head-P 的准确率
    # 对于 FedAvg, test_accuracies 是全局模型的准确率
    print(f"最终准确率: {final_acc:.4f}")
    if args.algorithm == 'fedrod':
        print("  (FedRoD: 这是 Head-P 的个性化准确率)")
    else:
        print("  (FedAvg: 这是全局模型的准确率)")

    # 尾部类别准确率
    if 'tail_accuracies' in training_history and training_history['tail_accuracies']:
        final_tail_acc = training_history['tail_accuracies'][-1]
        print(f"最终尾部类别准确率: {final_tail_acc:.4f}")
        if final_acc > 0:
            print(f"尾部性能差距: {final_acc - final_tail_acc:.4f}")
    else:
        print("提示: 未计算尾部类别准确率")

    # 层级错误
    if 'hierarchical_errors' in training_history and training_history['hierarchical_errors']:
        final_hier_err = training_history['hierarchical_errors'][-1]
        print(f"最终层级错误: {final_hier_err:.4f}")
    else:
        print("提示: 未计算层级错误")

    # IN-C准确率 (OOD泛化)
    if 'inc_accuracies' in training_history and training_history['inc_accuracies']:
        final_inc_acc = training_history['inc_accuracies'][-1]
        print(f"最终IN-C准确率: {final_inc_acc:.4f}")
        if final_acc > 0:
            print(f"泛化性能下降: {final_acc - final_inc_acc:.4f}")
    else:
        print("提示: 未计算IN-C准确率 (可能未提供IN-C数据)")

    # Near-OOD检测
    if 'near_auroc' in training_history and training_history['near_auroc']:
        final_near_auroc = training_history['near_auroc'][-1]
        print(f"最终Near-OOD AUROC: {final_near_auroc:.4f}")
    else:
        print("提示: 未计算Near-OOD AUROC (可能未提供Near-OOD数据)")

    # Far-OOD检测
    if 'far_auroc' in training_history and training_history['far_auroc']:
        final_far_auroc = training_history['far_auroc'][-1]
        print(f"最终Far-OOD AUROC: {final_far_auroc:.4f}")
    else:
        print("提示: 未计算Far-OOD AUROC (可能未提供Far-OOD数据)")

    # 数据点统计
    print(f"\n数据点统计:")
    print(f"  总通信轮次: {len(training_history['rounds']) if 'rounds' in training_history else 0}")
    print(f"  评估频率: 每 {args.eval_frequency} 轮评估一次")
    print(f"  保存频率: 每 {args.save_frequency} 轮保存一次检查点")
    print(f"  曲线数据点: {len(training_history['test_accuracies']) if training_history['test_accuracies'] else 0} 个")


if __name__ == "__main__":
    # [新增] 开启 cudnn benchmark
    torch.backends.cudnn.benchmark = True

    main()