import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any
# 引入评估工具函数
from eval_utils import evaluate_id_performance, evaluate_ood_detection

class FLServer:
    # 修改初始化，接收 foogd_module
    def __init__(self, global_model, foogd_module, device): 
        self.global_model = global_model
        self.foogd_module = foogd_module # 新增
        self.device = device
        self.global_model.to(device)
        if self.foogd_module:
            self.foogd_module.to(device)

    def get_global_parameters(self):
        """获取全局参数 (Model + FOOGD)"""
        global_params = {}
        # 1. Model Params
        for name, param in self.global_model.named_parameters():
            if 'head_p' not in name:
                global_params[f"model.{name}"] = param.data.clone()
        
        # 2. FOOGD Params
        if self.foogd_module:
            for name, param in self.foogd_module.named_parameters():
                global_params[f"foogd.{name}"] = param.data.clone()
                
        return global_params

    def set_global_parameters(self, params):
        """设置全局参数"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                key = f"model.{name}"
                if 'head_p' not in name and key in params:
                    param.data.copy_(params[key])
            
            if self.foogd_module:
                for name, param in self.foogd_module.named_parameters():
                    key = f"foogd.{name}"
                    if key in params:
                        param.data.copy_(params[key])

    def aggregate(self, updates, sample_sizes):
        """
        聚合客户端更新 (FedAvg 加权平均)
        
        Args:
            updates: 客户端参数更新列表
            sample_sizes: 客户端样本数量列表
            
        Returns:
            aggregated_params: 聚合后的全局参数
        """
        total_samples = sum(sample_sizes)
        
        # 以第一个客户端的更新为模板初始化
        aggregated_params = copy.deepcopy(updates[0])
        
        # 清零，准备累加
        for name in aggregated_params:
            aggregated_params[name] = torch.zeros_like(aggregated_params[name])
            
        # 加权累加
        for update, n_samples in zip(updates, sample_sizes):
            weight = n_samples / total_samples
            for name, param in update.items():
                if name in aggregated_params:
                    aggregated_params[name] += param * weight
                    
        return aggregated_params

    def _compute_ood_scores(self, data_loader):
        """
        修正：使用 Score Model 计算真正的 OOD 分数
        """
        self.global_model.eval()
        if self.foogd_module:
            self.foogd_module.eval()

        all_scores = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                _, _, features = self.global_model(data)

                if self.foogd_module:
                    # [修正] 必须先归一化，与训练时保持一致！
                    features_norm = F.normalize(features, p=2, dim=1)
                    # 使用归一化后的特征计算分数
                    _, _, scores = self.foogd_module(features_norm)
                else:
                    # 退化：使用特征范数
                    scores = torch.norm(features, dim=1)
                    
                all_scores.extend(scores.cpu().numpy())

        return np.array(all_scores)

    def evaluate_global_model(self, test_loader, near_ood_loader, far_ood_loader, inc_loader=None):
        """
        评估全局模型性能
        新增 inc_loader 参数用于测试 OOD 泛化能力
        """
        self.global_model.eval()
        if self.foogd_module:
            self.foogd_module.eval()
            
        metrics = {}

        # 1. 评估 ID (Clean) 准确率
        print("  评估 ID (Clean) ...")
        # 注意：这里需要传入 device
        id_metrics = evaluate_id_performance(self.global_model, test_loader, self.device)
        metrics['id_accuracy'] = id_metrics['accuracy']
        # 如果 evaluate_id_performance 没有返回 loss，这里设为 0 或自己计算
        metrics['id_loss'] = id_metrics['loss']

        # 2. [新增] 评估 IN-C (Corrupted) 准确率 -> 验证 SAG 效果
        if inc_loader:
            print("  评估 IN-C (OOD Generalization) ...")
            inc_metrics = evaluate_id_performance(self.global_model, inc_loader, self.device)
            metrics['inc_accuracy'] = inc_metrics['accuracy']

        # 3. 评估 OOD 检测 (Near & Far) -> 验证 SM3D 效果
        if near_ood_loader:
            print("  评估 Near-OOD 检测 ...")
            near_metrics = evaluate_ood_detection(
                self.global_model, self.foogd_module, test_loader, near_ood_loader, self.device
            )
            metrics['near_auroc'] = near_metrics['auroc']
            
        if far_ood_loader:
            print("  评估 Far-OOD 检测 ...")
            far_metrics = evaluate_ood_detection(
                self.global_model, self.foogd_module, test_loader, far_ood_loader, self.device
            )
            metrics['far_auroc'] = far_metrics['auroc']

        return metrics

if __name__ == "__main__":
    # 测试服务端
    print("测试联邦学习服务端...")

    # 创建模型
    from models import create_fedrod_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model, foogd_module = create_fedrod_model()
    server = FLServer(model, foogd_module, device)

    # 测试参数获取和设置
    print("\n测试参数管理...")
    global_params = server.get_global_parameters()
    print(f"全局参数数量: {len(global_params)}")

    # 测试聚合
    print("\n测试参数聚合...")
    client_updates = [
        {name: torch.randn_like(param) for name, param in global_params.items()},
        {name: torch.randn_like(param) for name, param in global_params.items()},
        {name: torch.randn_like(param) for name, param in global_params.items()}
    ]
    client_sample_sizes = [100, 150, 200]

    aggregated_params = server.aggregate(client_updates, client_sample_sizes)
    print(f"聚合参数数量: {len(aggregated_params)}")

    # 测试评估
    print("\n测试模型评估...")
    from data_utils import create_federated_loaders

    data_root = "./data"
    # 注意：这里使用 try-except 块，以防没有数据时报错，仅作测试逻辑演示
    try:
        _, test_loader, near_ood_loader, far_ood_loader, _ = create_federated_loaders(
            data_root, n_clients=3, batch_size=4, image_size=224
        )

        metrics = server.evaluate_global_model(
            test_loader, near_ood_loader, far_ood_loader, None
        )

        print("评估指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"跳过数据评估测试（可能缺少数据）: {e}")

    print("服务端测试完成!")