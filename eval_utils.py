#!/usr/bin/env python3
"""
评估工具模块
用于评估pFL-FOOGD模型的性能

作者: Claude Code
日期: 2025-11-22
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch.nn.functional as F

def evaluate_id_performance(model, data_loader, device, num_classes=54):
    """
    评估ID数据上的分类性能

    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        num_classes: 类别数量

    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    total_loss = 0.0 # 新增
    total_samples = 0 # 新增
    all_preds = []
    all_targets = []
    all_logits_g = []
    all_logits_p = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            logits_g, logits_p, _ = model(data)

# 计算 Loss (和 client.py 保持一致，取平均或只看 head_g)
            loss = F.cross_entropy(logits_g, targets) 
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # 使用通用头进行预测
            _, preds = torch.max(logits_g, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits_g.extend(logits_g.cpu().numpy())
            all_logits_p.extend(logits_p.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits_g = np.array(all_logits_g)
    all_logits_p = np.array(all_logits_p)

    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)

    # 计算每个类别的准确率
    class_accuracy = {}
    for class_idx in range(num_classes):
        class_mask = all_targets == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(all_targets[class_mask], all_preds[class_mask])
            class_accuracy[class_idx] = class_acc

    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

    metrics = {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets,
        'logits_g': all_logits_g,
        'logits_p': all_logits_p,
        'loss': total_loss / total_samples if total_samples > 0 else 0
    }

    return metrics


def evaluate_ood_detection(model, foogd_module, id_loader, ood_loader, device):
    """
    评估OOD检测性能

    Args:
        model: 模型
        foogd_module: FOOGD模块
        id_loader: ID数据加载器
        ood_loader: OOD数据加载器
        device: 设备

    Returns:
        metrics: OOD检测指标字典
    """
    model.eval()
    if foogd_module:
        foogd_module.eval()

    # 收集ID和OOD分数
    id_scores = compute_ood_scores(model, foogd_module, id_loader, device)
    ood_scores = compute_ood_scores(model, foogd_module, ood_loader, device)

    # 合并分数和标签
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

    # 计算OOD检测指标
    auroc = roc_auc_score(labels, scores)

    # 计算AUPR
    aupr_in = average_precision_score(labels, scores)
    aupr_out = average_precision_score(1 - labels, -scores)

    # 计算FPR@95
    fpr, tpr, thresholds = roc_curve(labels, scores)
    tpr_target = 0.95
    idx = np.argmin(np.abs(tpr - tpr_target))
    fpr95 = fpr[idx]

    # 计算检测错误率
    detection_error = compute_detection_error(labels, scores)

    metrics = {
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out,
        'fpr95': fpr95,
        'detection_error': detection_error,
        'id_scores': id_scores,
        'ood_scores': ood_scores,
        'labels': labels
    }

    return metrics


def compute_ood_scores(model, foogd_module, data_loader, device):
    """
    计算OOD分数

    Args:
        model: 模型
        foogd_module: FOOGD模块
        data_loader: 数据加载器
        device: 设备

    Returns:
        ood_scores: OOD分数数组
    """
    all_scores = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)

            _, _, features = model(data)

            if foogd_module:
                # [修正] 必须先归一化，与训练时保持一致！
                features_norm = F.normalize(features, p=2, dim=1)
                # 使用归一化后的特征计算分数
                _, _, scores = foogd_module(features_norm)
            else:
                # 如果没有 FOOGD，使用特征范数
                scores = torch.norm(features, dim=1)

            all_scores.extend(scores.cpu().numpy())

    return np.array(all_scores)


def compute_detection_error(labels, scores):
    """
    计算检测错误率

    Args:
        labels: 真实标签
        scores: OOD分数

    Returns:
        detection_error: 检测错误率
    """
    # 找到最优阈值
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    detection_errors = (fpr + fnr) / 2
    min_error_idx = np.argmin(detection_errors)

    return detection_errors[min_error_idx]


def plot_ood_detection_results(id_scores, ood_scores, output_path):
    """
    绘制OOD检测结果

    Args:
        id_scores: ID样本的OOD分数
        ood_scores: OOD样本的OOD分数
        output_path: 输出路径
    """
    plt.figure(figsize=(15, 5))

    # 分数分布
    plt.subplot(1, 3, 1)
    plt.hist(id_scores, bins=50, alpha=0.7, label='ID', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.7, label='OOD', density=True)
    plt.xlabel('OOD Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('OOD Score Distribution')

    # ROC曲线
    plt.subplot(1, 3, 2)
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    plt.plot(fpr, tpr, 'b-', label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend()
    plt.title('ROC Curve')

    # Precision-Recall曲线
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = average_precision_score(labels, scores)

    plt.plot(recall, precision, 'g-', label=f'AUPR = {aupr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        output_path: 输出路径
    """
    plt.figure(figsize=(12, 10))

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names[:10], yticklabels=class_names[:10])  # 只显示前10个类别

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_evaluation_report(model, foogd_module, test_loader, near_ood_loader,
                              far_ood_loader, device, output_dir):
    """
    生成完整的评估报告

    Args:
        model: 模型
        foogd_module: FOOGD模块
        test_loader: 测试数据加载器
        near_ood_loader: Near-OOD数据加载器
        far_ood_loader: Far-OOD数据加载器
        device: 设备
        output_dir: 输出目录

    Returns:
        report: 评估报告字典
    """
    os.makedirs(output_dir, exist_ok=True)

    report = {}

    # 1. ID分类性能评估
    print("评估ID分类性能...")
    id_metrics = evaluate_id_performance(model, test_loader, device)
    report['id_classification'] = id_metrics

    # 绘制混淆矩阵
    plot_confusion_matrix(
        id_metrics['confusion_matrix'],
        [f'Class_{i}' for i in range(54)],
        os.path.join(output_dir, 'confusion_matrix.png')
    )

    # 2. Near-OOD检测评估
    print("评估Near-OOD检测性能...")
    if near_ood_loader is not None:
        near_ood_metrics = evaluate_ood_detection(
            model, foogd_module, test_loader, near_ood_loader, device
        )
        report['near_ood_detection'] = near_ood_metrics

        plot_ood_detection_results(
            near_ood_metrics['id_scores'],
            near_ood_metrics['ood_scores'],
            os.path.join(output_dir, 'near_ood_detection.png')
        )

    # 3. Far-OOD检测评估
    print("评估Far-OOD检测性能...")
    if far_ood_loader is not None:
        far_ood_metrics = evaluate_ood_detection(
            model, foogd_module, test_loader, far_ood_loader, device
        )
        report['far_ood_detection'] = far_ood_metrics

        plot_ood_detection_results(
            far_ood_metrics['id_scores'],
            far_ood_metrics['ood_scores'],
            os.path.join(output_dir, 'far_ood_detection.png')
        )

    # 保存报告
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        import json
        # 转换numpy数组为列表
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                serializable_report[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        serializable_report[key][sub_key] = sub_value.tolist()
                    else:
                        serializable_report[key][sub_key] = sub_value
            else:
                serializable_report[key] = value

        json.dump(serializable_report, f, indent=2)

    print(f"评估报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    # 测试评估工具
    print("测试评估工具...")

    # 创建虚拟数据
    id_scores = np.random.normal(0.5, 0.2, 1000)
    ood_scores = np.random.normal(0.8, 0.3, 1000)

    # 测试绘图函数
    plot_ood_detection_results(id_scores, ood_scores, "test_ood_detection.png")
    print("OOD检测结果图已保存: test_ood_detection.png")

    # 测试混淆矩阵
    cm = np.random.randint(0, 100, (10, 10))
    plot_confusion_matrix(cm, [f'Class_{i}' for i in range(10)], "test_confusion_matrix.png")
    print("混淆矩阵图已保存: test_confusion_matrix.png")

    print("评估工具测试完成!")