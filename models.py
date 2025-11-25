#!/usr/bin/env python3
"""
模型定义模块 - 包含FedRoD模型架构和FOOGD组件
基于项目工作文档中的架构设计

作者: Claude Code
日期: 2025-11-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights


class DenseNetBackbone(nn.Module):
    """DenseNet骨干网络 - 特征提取器"""

    def __init__(self, model_type='densenet169', pretrained=True):
        """
        初始化DenseNet骨干网络

        Args:
            model_type: 模型类型 ('densenet121', 'densenet169')
            pretrained: 是否使用预训练权重
        """
        super(DenseNetBackbone, self).__init__()

        # 根据 pretrained 参数决定使用什么权重
        weights_121 = DenseNet121_Weights.DEFAULT if pretrained else None
        weights_169 = DenseNet169_Weights.DEFAULT if pretrained else None

        if model_type == 'densenet121':
            self.backbone = models.densenet121(weights=weights_121)
            self.feature_dim = 1024
        elif model_type == 'densenet169':
            self.backbone = models.densenet169(weights=weights_169)
            self.feature_dim = 1664
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # 移除最后的分类层
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [batch_size, 3, height, width]

        Returns:
            features: 特征向量 [batch_size, feature_dim]
        """
        features = self.backbone(x)
        return features


class FedRoD_Model(nn.Module):
    """FedRoD模型 - 包含通用头和个性化头"""

    def __init__(self, backbone, num_classes=54, hidden_dim=512):
        """
        初始化FedRoD模型

        Args:
            backbone: 骨干网络
            num_classes: 类别数量
            hidden_dim: 隐藏层维度
        """
        super(FedRoD_Model, self).__init__()

        self.backbone = backbone
        self.feature_dim = backbone.feature_dim

        # 通用头 (Head_G) - 服务器聚合
        self.head_g = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        # 个性化头 (Head_P) - 客户端本地保持
        self.head_p = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像

        Returns:
            logits_g: 通用头输出
            logits_p: 个性化头输出
            features: 特征向量
        """
        features = self.backbone(x)
        logits_g = self.head_g(features)
        logits_p = self.head_p(features)

        return logits_g, logits_p, features


class ScoreModel(nn.Module):
    """评分模型 - 用于SM3D OOD检测"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        """
        初始化评分模型

        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
        """
        super(ScoreModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # 输出层 - 输出与输入相同维度的梯度
        layers.append(nn.Linear(prev_dim, input_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征

        Returns:
            score: 评分向量 [batch_size, input_dim]
        """
        return self.network(x)


class FOOGD_Module(nn.Module):
    """
    FOOGD模块 - 集成SAG和SM3D
    """

    def __init__(self, feature_dim, num_classes=54):
        super(FOOGD_Module, self).__init__()
        self.feature_dim = feature_dim
        self.score_model = ScoreModel(feature_dim)

    def rbf_kernel_matrix(self, X, Y, sigma=None):
        """计算RBF核矩阵 K(X, Y)"""
        if sigma is None:
            dists = torch.cdist(X, Y)
            sigma = torch.median(dists).detach()
        gamma = 1.0 / (2 * sigma**2 + 1e-8)
        K_XY = torch.exp(-gamma * torch.cdist(X, Y)**2)
        return K_XY, sigma

    def langevin_dynamic_sampling(self, batch_size, device, step_size=0.01, n_steps=10):
        z_k = torch.randn(batch_size, self.feature_dim).to(device)
        z_k.requires_grad = True
        for _ in range(n_steps):
            noise = torch.randn_like(z_k)
            with torch.no_grad():
                score = self.score_model(z_k)
            z_k = z_k + (step_size / 2) * score + torch.sqrt(torch.tensor(step_size)) * noise
        return z_k.detach()

    def compute_mmd_loss(self, z_real, z_gen):
        K_xx, _ = self.rbf_kernel_matrix(z_real, z_real)
        K_yy, _ = self.rbf_kernel_matrix(z_gen, z_gen)
        K_xy, _ = self.rbf_kernel_matrix(z_real, z_gen)
        loss_mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return loss_mmd

    def compute_sm3d_loss(self, features, lambda_m=0.5):
        # 1. Detach features (保持不变)
        features_fixed = features.detach()
        batch_size = features_fixed.size(0)

        # --- Part A: Denoising Score Matching (DSM) ---
        sigma = 0.1 # 建议提取为类变量或参数
        noise = torch.randn_like(features_fixed) * sigma
        noisy_features = features_fixed + noise

        # 预测 Score
        score_pred = self.score_model(noisy_features)

        # 真实 Score (目标)
        # target = -noise / sigma^2
        score_true = -noise / (sigma ** 2)

        # 1. 计算 Mean MSE (数值约为 50)
        raw_dsm_loss = 0.5 * F.mse_loss(score_pred, score_true, reduction='mean')

        # 2. [修正] 只乘 sigma^2，不要乘 feature_dim
        # 这样可以将 Loss 缩放到 0.5 左右 (50 * 0.01 = 0.5)
        # 这与 Classification Loss (0.5~2.0) 是完全匹配的！
        l_dsm = raw_dsm_loss * (sigma ** 2)

        # --- Part B: MMD (保持不变) ---
        z_gen = self.langevin_dynamic_sampling(batch_size, features.device)
        l_mmd = self.compute_mmd_loss(features_fixed, z_gen)

        # 总损失
        total_loss = (1 - lambda_m) * l_dsm + lambda_m * l_mmd

        return total_loss

    def compute_ksd_loss(self, z, z_aug):
        """
        修正版：计算 KSD(p(z), q(z_aug))
        目标：优化 Feature Extractor，使得 z_aug 落在 p(z) 的高密度区域
        """

        # 1. 关键修改：操作对象必须是【增强特征 z_aug】
        features = z_aug 

        # 2. Score Model 作为 Critic，不需要更新它，所以 detach
        # 但它需要对 features (z_aug) 进行打分
        # 注意：论文 Eq. 11 中 s_theta(z_hat) 是 score function
        scores = self.score_model(features).detach()

        #    3. 计算 Kernel (在增强特征之间计算)
        # features (即 z_aug) 需要保留梯度，以便反向传播更新 Backbone
        K_xx, sigma = self.rbf_kernel_matrix(features, features)

        # --- 以下计算逻辑保持不变，但现在的 'features' 和 'scores' 对应的是 z_aug ---

        # Term 1: s(x)^T s(x') * k(x,x')
        term1 = torch.matmul(scores, scores.t()) * K_xx

        # 辅助变量: (x_i - x_j)
        # [B, B, Dim]
        X_diff = features.unsqueeze(1) - features.unsqueeze(0)

        # 修正后的 Term 2
        # scores: [Batch(i), Dim(d)] -> 'id'
        # X_diff: [Batch(i), Batch(j), Dim(d)] -> 'ijd'
        # 结果: [Batch(i), Batch(j)] -> 'ij'
        # 物理含义: 计算当前样本 i 的 score 与差向量 (x_i - x_j) 的点积
        term2 = torch.einsum('id, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)

        # 修正后的 Term 3
        # scores: [Batch(j), Dim(d)] -> 'jd' (注意这里取的是 j，代表对方样本)
        # X_diff: [Batch(i), Batch(j), Dim(d)] -> 'ijd'
        # 结果: [Batch(i), Batch(j)] -> 'ij'
        # 物理含义: 计算对方样本 j 的 score 与差向量 (x_i - x_j) 的点积
        term3 = -torch.einsum('jd, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)
    
        # Term 4: trace(grad_x grad_x' k)
        dim = features.size(1)
        sq_dist = torch.sum(X_diff**2, dim=2)
        term4 = (dim / (sigma**2) - sq_dist / (sigma**4)) * K_xx
        
        # 最终取平均
        ksd = (term1 + term2 + term3 + term4).mean()

        return ksd

    def forward(self, features, features_aug=None):
        # 1. OOD Score (测试用) - 使用正范数作为异常分数 (Anomaly Score)
        # 范数越大 -> 梯度越大 -> 密度越低 -> 越可能是 OOD
        ood_scores = torch.norm(self.score_model(features), dim=1)

        # 2. SM Loss (训练 Score Model)
        sm_loss = self.compute_sm3d_loss(features)

        # 3. KSD Loss (训练 Backbone)
        if features_aug is not None:
            ksd_loss = self.compute_ksd_loss(features, features_aug)
        else:
            ksd_loss = torch.tensor(0.0, device=features.device)

        return ksd_loss, sm_loss, ood_scores


def create_fedrod_model(model_type='densenet169', num_classes=54, use_foogd=True):
    """
    创建FedRoD模型

    Args:
        model_type: 骨干网络类型
        num_classes: 类别数量
        use_foogd: 是否使用FOOGD模块

    Returns:
        model: FedRoD模型
        foogd_module: FOOGD模块（如果使用）
    """
    # 创建骨干网络
    backbone = DenseNetBackbone(model_type=model_type, pretrained=True)

    # 创建FedRoD模型
    model = FedRoD_Model(backbone, num_classes=num_classes)

    # 创建FOOGD模块
    foogd_module = None
    if use_foogd:
        foogd_module = FOOGD_Module(backbone.feature_dim, num_classes)

    return model, foogd_module


if __name__ == "__main__":
    # 测试模型
    print("测试FedRoD模型...")

    # 创建模型
    model, foogd_module = create_fedrod_model()

    # 测试前向传播
    dummy_input = torch.randn(4, 3, 224, 224)
    logits_g, logits_p, features = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"通用头输出尺寸: {logits_g.shape}")
    print(f"个性化头输出尺寸: {logits_p.shape}")
    print(f"特征向量尺寸: {features.shape}")

    # 测试FOOGD模块
    if foogd_module:
        ksd_loss, sm_loss, ood_scores = foogd_module(features)
        print(f"KSD损失: {ksd_loss.item():.4f}")
        print(f"评分匹配损失: {sm_loss.item():.4f}")
        print(f"OOD分数尺寸: {ood_scores.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")

    if foogd_module:
        foogd_params = sum(p.numel() for p in foogd_module.parameters())
        print(f"FOOGD模块参数: {foogd_params:,}")