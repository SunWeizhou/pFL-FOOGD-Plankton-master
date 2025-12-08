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
        """获取全局参数 (Model + FOOGD) - 修复版"""
        # [修复] 返回带前缀的参数，与 client.set_generic_parameters 保持一致
        params = {}
        model_state = self.global_model.state_dict()
        for key, value in model_state.items():
            params[f"model.{key}"] = value.clone()
        if self.foogd_module:
            foogd_state = self.foogd_module.state_dict()
            for key, value in foogd_state.items():
                params[f"foogd.{key}"] = value.clone()
        return params

    def set_global_parameters(self, params):
        """设置全局参数 - 修复版：支持带前缀的参数拆解"""

        # 1. 初始化两个空字典
        model_params = {}
        foogd_params = {}

        # 2. 遍历聚合后的参数，根据前缀拆分
        for key, value in params.items():
            if key.startswith("model."):
                # 去掉 "model." 前缀，还原为模型能识别的键名
                new_key = key.replace("model.", "")
                model_params[new_key] = value
            elif key.startswith("foogd."):
                # 去掉 "foogd." 前缀
                new_key = key.replace("foogd.", "")
                foogd_params[new_key] = value
            else:
                # 兼容旧逻辑（如果某个参数没有前缀，尝试直接归类给model）
                model_params[key] = value

        # 3. 加载主模型参数
        # strict=False 仍然很重要，以防有些缓冲变量不匹配，但核心权重现在能对上了
        self.global_model.load_state_dict(model_params, strict=False)

        # 4. 加载 FOOGD 参数 [关键修复点]
        if self.foogd_module and foogd_params:
            self.foogd_module.load_state_dict(foogd_params, strict=False)

        print(f"  [Server] Updated Global Model with {len(model_params)} params")
        if self.foogd_module:
             print(f"  [Server] Updated FOOGD Module with {len(foogd_params)} params")

    def aggregate(self, updates, sample_sizes):
        """
        聚合函数 - [已修正] 恢复 BN 统计量的聚合
        """
        total_samples = sum(sample_sizes)
        # 1. 初始化聚合参数为第一个客户端的参数副本
        aggregated_params = copy.deepcopy(updates[0])

        for key in aggregated_params.keys():
            # [关键修改]：
            # 我们只跳过 'num_batches_tracked' (它是整数，记录训练步数，不需要平均)
            # 删除了之前过滤 'running_mean' 和 'running_var' 的逻辑
            if 'num_batches_tracked' in key:
                continue

            # 2. 准备进行加权平均
            # 先将当前参数置为 0
            # 注意：这里我们假设参数都是 Tensor 类型
            aggregated_params[key] = torch.zeros_like(aggregated_params[key], dtype=torch.float)

            for update, n_samples in zip(updates, sample_sizes):
                weight = n_samples / total_samples

                param_data = update[key]

                # 确保参与计算的数据是 float 类型
                if param_data.dtype != torch.float:
                    param_data = param_data.float()

                # 加权累加
                aggregated_params[key] += param_data * weight

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
    client_updates = []
    for _ in range(3):
        client_update = {}
        for name, param in global_params.items():
            # 只对浮点类型参数生成随机数
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                client_update[name] = torch.randn_like(param)
            else:
                # 对于整数类型参数，保持原值
                client_update[name] = param.clone()
        client_updates.append(client_update)
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