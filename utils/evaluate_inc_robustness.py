#!/usr/bin/env python3
"""
IN-C (ImageNet-C 风格) 鲁棒性评估脚本
专门用于评估 pFL-FOOGD 模型的域泛化能力

功能:
1. 加载最佳模型 (best_model.pth)
2. 在 ID 测试集上计算基准准确率
3. 遍历所有腐蚀类型 (Blur, Noise, Brightness) 和严重程度 (1-5)
4. 生成详细的鲁棒性分析报告

作者: Gemini based on user's code
日期: 2025-12-11
"""

import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入你项目中的模块
from models import create_fedrod_model
from data_utils import PlanktonDataset, Corruptions
from server import FLServer

def parse_args():
    parser = argparse.ArgumentParser(description='评估 pFL-FOOGD 模型的 IN-C 鲁棒性')

    # 路径参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径 (例如: experiments/.../best_model.pth)')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                       help='数据集根目录')

    # 模型配置 (需要与训练时一致)
    parser.add_argument('--model_type', type=str, default='densenet121',
                       choices=['densenet121', 'densenet169'],
                       help='骨干网络类型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--image_size', type=int, default=299, help='图像尺寸')
    parser.add_argument('--use_foogd', action='store_true', help='是否加载 FOOGD 模块')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def get_custom_inc_loader(data_root, batch_size, image_size, corruption_func, severity):
    """
    创建自定义的 IN-C 加载器，支持动态指定腐蚀函数
    """
    # 定义包含特定腐蚀的变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # 插入动态腐蚀操作
        transforms.Lambda(lambda x: corruption_func(x, severity=severity)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 使用 ID 测试集数据 (mode='test')
    dataset = PlanktonDataset(data_root, transform=transform, mode='test')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return loader

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== 开始 IN-C 鲁棒性评估 ===")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {device}")

    # 1. 加载模型结构
    print("\n[1/4] 初始化模型...")
    global_model, foogd_module = create_fedrod_model(
        model_type=args.model_type,
        num_classes=54,
        use_foogd=args.use_foogd # 即使不用FOOGD做推理，加载权重也需要匹配key
    )
    global_model.to(device)
    if foogd_module:
        foogd_module.to(device)

    # 2. 加载权重
    print(f"[2/4] 加载权重...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)

    # 兼容处理: 检查key是否包含 'global_model_state_dict' (best_model格式) 或是直接的 state_dict
    if 'global_model_state_dict' in checkpoint:
        print(f"  -> 识别为完整检查点 (Round {checkpoint.get('round', '?')})")
        global_model.load_state_dict(checkpoint['global_model_state_dict'])
        if foogd_module and 'foogd_state_dict' in checkpoint and checkpoint['foogd_state_dict']:
            foogd_module.load_state_dict(checkpoint['foogd_state_dict'])
    else:
        print("  -> 识别为纯权重文件")
        # 尝试去掉可能存在的 module. 前缀
        state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_key] = v
        global_model.load_state_dict(new_state_dict, strict=False)

    # 创建 Server 实例以便复用评估逻辑
    server = FLServer(global_model, foogd_module, device)

    # 3. 评估基准性能 (Clean Accuracy)
    print("\n[3/4] 评估 Clean Data 基准性能...")
    # 使用 data_utils 默认的 test_loader (无腐蚀)
    from data_utils import create_federated_loaders
    _, test_loader, _, _, _ = create_federated_loaders(
        args.data_root, n_clients=1, batch_size=args.batch_size, image_size=args.image_size
    )

    clean_metrics = server._compute_scores_and_metrics(test_loader)
    clean_acc = clean_metrics['accuracy']
    print(f"  ★ Clean Accuracy: {clean_acc:.2%}")

    # 4. 评估 IN-C 鲁棒性
    print("\n[4/4] 扫描所有腐蚀类型和强度...")

    # 定义要测试的腐蚀类型映射 (名称 -> 函数)
    corruptions_map = {
        'Gaussian Blur': Corruptions.gaussian_blur,
        'Gaussian Noise': Corruptions.gaussian_noise,
        'Brightness': Corruptions.brightness
    }

    results = {}

    # 打印表头
    print(f"\n{'Corruption Type':<20} | {'Severity':<10} | {'Accuracy':<10} | {'Drop':<10}")
    print("-" * 60)

    for c_name, c_func in corruptions_map.items():
        results[c_name] = []
        for severity in range(1, 6): # 1 到 5
            # 创建对应的 Loader
            inc_loader = get_custom_inc_loader(
                args.data_root,
                args.batch_size,
                args.image_size,
                c_func,
                severity
            )

            # 评估
            metrics = server._compute_scores_and_metrics(inc_loader)
            acc = metrics['accuracy']
            drop = clean_acc - acc

            results[c_name].append(acc)

            print(f"{c_name:<20} | {severity:<10} | {acc:.2%}    | -{drop:.2%}")

    # 5. 打印最终汇总表格 (Markdown 格式)
    print("\n\n=== �� IN-C 鲁棒性评估报告 ===")
    print("可以将以下表格直接复制到你的 Markdown 报告中：\n")

    print("| Corruption Type | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 | **Mean Acc** | **mCE (Drop)** |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    avg_acc_all = []

    for c_name, accs in results.items():
        mean_acc = np.mean(accs)
        mean_drop = clean_acc - mean_acc
        avg_acc_all.extend(accs)

        row_str = f"| {c_name} | "
        for acc in accs:
            row_str += f"{acc:.2%} | "
        row_str += f"**{mean_acc:.2%}** | -{mean_drop:.2%} |"
        print(row_str)

    total_mean_acc = np.mean(avg_acc_all)
    print(f"| **Overall** | | | | | | **{total_mean_acc:.2%}** | **-{clean_acc - total_mean_acc:.2%}** |")

    print(f"\n基准 Clean Accuracy: {clean_acc:.2%}")

if __name__ == "__main__":
    main()