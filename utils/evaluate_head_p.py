#!/usr/bin/env python3
"""
pFL-FOOGD 个性化性能评估脚本 (独立版)
功能：加载已保存的 best_model.pth，在同分布(Non-IID)测试集上评估 Head-P 与 Head-G。
特点：不依赖对原代码的修改，自包含所有必要逻辑。

作者: Claude Code
日期: 2025-12-11
"""

import os
import torch
import json
import numpy as np
import argparse
import copy
from torch.utils.data import DataLoader, Subset

# 直接导入原有模块 (只读，不修改)
from models import create_fedrod_model
from client import FLClient
from data_utils import PlanktonDataset, get_transforms

def set_seed(seed):
    """复现随机环境"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def partition_test_set(dataset, n_clients, alpha, n_classes=54, seed=None):
    """
    [核心逻辑] 在不修改 data_utils.py 的情况下，在这里实现确定性的数据划分
    """
    # 1. 临时设置随机状态以复现分布比例
    if seed is not None:
        np.random.seed(seed)
        
    # 2. 生成每个类别的分布比例矩阵 [n_classes, n_clients]
    # 这步模拟了 data_utils.partition_data 中的 dirichlet 采样过程
    proportions_matrix = []
    for _ in range(n_classes):
        p = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions_matrix.append(p)
    
    # 3. 开始划分测试集数据
    class_indices = {i: [] for i in range(n_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label >= 0 and label < n_classes:
            class_indices[label].append(idx)

    client_indices = [[] for _ in range(n_clients)]

    # 使用刚才生成的比例矩阵进行分配
    for class_idx in range(n_classes):
        class_data = class_indices[class_idx]
        # 注意：这里打乱测试集样本顺序是没关系的，只要总数比例对就行
        # 为了严谨，我们使用同样的 seed 再次打乱
        np.random.shuffle(class_data)
        
        probs = proportions_matrix[class_idx]
        counts = (probs * len(class_data)).astype(int)
        counts[-1] = len(class_data) - np.sum(counts[:-1])

        start = 0
        for client_id in range(n_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_data[start:end])
            start = end

    return client_indices

def evaluate_single_run(exp_path, data_root, device):
    """评估单个实验目录"""
    config_path = os.path.join(exp_path, "config.json")
    model_path = os.path.join(exp_path, "best_model.pth")
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None

    print(f"\n>> 正在评估: {os.path.basename(exp_path)}")
    
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    alpha = config['alpha']
    seed = config['seed']
    n_clients = config['n_clients']
    img_size = config.get('image_size', 224) # 兼容旧配置
    
    print(f"   配置: Alpha={alpha} | Seed={seed} | FOOGD={config['use_foogd']}")

    # 2. 准备同分布测试集
    # 这一步很关键：我们用同样的 Seed 和 Alpha 重新切分测试集
    # 让 Client 1 的测试集分布 = Client 1 的训练集分布
    set_seed(seed) 
    _, test_transform = get_transforms(img_size)
    test_dataset = PlanktonDataset(data_root, transform=test_transform, mode='test')
    
    # 调用本地实现的划分函数
    client_test_indices = partition_test_set(test_dataset, n_clients, alpha, seed=seed)
    
    # 3. 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    if 'client_states' not in checkpoint:
        print("   ❌ 错误: 缺少 client_states，无法评估个性化模型")
        return None
    
    client_states = checkpoint['client_states']
    
    # 4. 初始化模型骨架
    temp_model, temp_foogd = create_fedrod_model(
        model_type=config['model_type'],
        num_classes=54,
        use_foogd=config['use_foogd']
    )
    temp_model = temp_model.to(device)
    if temp_foogd: temp_foogd = temp_foogd.to(device)

    # 5. 评估循环
    results = {'acc_g': [], 'acc_p': [], 'gain': []}
    
    print(f"   {'Client':<6} | {'Test Size':<10} | {'Head-G':<8} | {'Head-P':<8} | {'Gain':<8}")
    print("   " + "-"*50)
    
    for i in range(n_clients):
        # A. 构建 DataLoader
        indices = client_test_indices[i]
        if len(indices) == 0:
            continue
            
        loader = DataLoader(
            Subset(test_dataset, indices),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # B. 加载参数 (Head-P 就在这里面)
        if i in client_states:
            temp_model.load_state_dict(client_states[i], strict=False)
        
        # C. 推理
        client = FLClient(i, temp_model, temp_foogd, None, device)
        metrics = client.evaluate(loader)
        
        gain = metrics['acc_p'] - metrics['acc_g']
        results['acc_g'].append(metrics['acc_g'])
        results['acc_p'].append(metrics['acc_p'])
        results['gain'].append(gain)
        
        print(f"   {i:<6} | {len(indices):<10} | {metrics['acc_g']:.4f}   | {metrics['acc_p']:.4f}   | {gain:+.4f}")

    avg_g = np.mean(results['acc_g'])
    avg_p = np.mean(results['acc_p'])
    avg_gain = np.mean(results['gain'])
    
    print("   " + "-"*50)
    print(f"   AVG    | -          | {avg_g:.4f}   | {avg_p:.4f}   | {avg_gain:+.4f}")
    
    return {
        'name': os.path.basename(exp_path),
        'alpha': alpha,
        'foogd': config['use_foogd'],
        'head_p': avg_p,
        'head_g': avg_g,
        'gain': avg_gain
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_dir', default='./experiments')
    parser.add_argument('--data_root', default='./Plankton_OOD_Dataset')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 扫描目录
    summary_list = []
    if os.path.exists(args.experiments_dir):
        for exp_name in sorted(os.listdir(args.experiments_dir)):
            exp_path = os.path.join(args.experiments_dir, exp_name)
            if not os.path.isdir(exp_path): continue
            
            # 找最新的 timestamp 文件夹
            subdirs = sorted([d for d in os.listdir(exp_path) if d.startswith('experiment_')])
            if subdirs:
                full_path = os.path.join(exp_path, subdirs[-1]) # 取最新
                res = evaluate_single_run(full_path, args.data_root, device)
                if res: summary_list.append(res)

    # 最终汇总
    print("\n" + "="*80)
    print("FINAL SUMMARY (Personalization Performance)")
    print("="*80)
    print(f"{'Experiment':<25} | {'Alpha':<5} | {'FOOGD':<5} | {'Head-P':<8} | {'Head-G':<8} | {'Gain'}")
    print("-" * 80)
    for item in summary_list:
        foogd = "Yes" if item['foogd'] else "No"
        print(f"{item['name']:<25} | {item['alpha']:<5} | {foogd:<5} | {item['head_p']:.4f}   | {item['head_g']:.4f}   | {item['gain']:+.4f}")
    print("="*80)

if __name__ == "__main__":
    main()