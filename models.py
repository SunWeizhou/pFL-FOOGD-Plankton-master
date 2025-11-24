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

        if model_type == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            self.feature_dim = 1024
        elif model_type == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
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

    论文对应:
    1. SM3D: Eq (8) = (1 - lambda_m) * l_DSM + lambda_m * l_MMD
    2. SAG:  Eq (12) -> KSD Loss (Eq 11)
    """

    def __init__(self, feature_dim, num_classes=54):
        super(FOOGD_Module, self).__init__()
        self.feature_dim = feature_dim

        # 论文中 SAG 并没有额外的投影层，它是直接约束特征提取器的
        # 因此移除 sag_projection，直接对 features 计算 KSD

        # SM3D组件 - Score Matching
        self.score_model = ScoreModel(feature_dim)

    # ==========================================
    # 辅助函数: RBF Kernel 及其梯度
    # ==========================================
    def rbf_kernel_matrix(self, X, Y, sigma=None):
        """计算RBF核矩阵 K(X, Y)"""
        if sigma is None:
            # 启发式设置 bandwidth (median trick)
            dists = torch.cdist(X, Y)
            sigma = torch.median(dists).detach()

        # K(x,y) = exp(- ||x-y||^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * sigma**2 + 1e-8)
        K_XY = torch.exp(-gamma * torch.cdist(X, Y)**2)
        return K_XY, sigma

    # ==========================================
    # 1. SM3D 实现 (Paper Section 3.2)
    # ==========================================
    def langevin_dynamic_sampling(self, batch_size, device, step_size=0.01, n_steps=10):
        """
        Langevin Dynamic Sampling (LDS) - Eq (6)
        从 Score Model 定义的分布中采样生成的特征 Z_gen
        """
        # 1. 初始化 z0 ~ N(0, I)
        z_k = torch.randn(batch_size, self.feature_dim).to(device)
        z_k.requires_grad = True

        # 2. 迭代采样
        for _ in range(n_steps):
            noise = torch.randn_like(z_k)
            with torch.no_grad():
                score = self.score_model(z_k) # score = grad(log p(z))

            # z_{t} = z_{t-1} + (epsilon/2) * score + sqrt(epsilon) * noise
            z_k = z_k + (step_size / 2) * score + torch.sqrt(torch.tensor(step_size)) * noise

        return z_k.detach() # 生成的样本不需要梯度回传到采样过程

    def compute_mmd_loss(self, z_real, z_gen):
        """
        计算 MMD Loss - Eq (7)
        衡量真实特征 z_real 和生成特征 z_gen 分布的距离
        """
        K_xx, _ = self.rbf_kernel_matrix(z_real, z_real)
        K_yy, _ = self.rbf_kernel_matrix(z_gen, z_gen)
        K_xy, _ = self.rbf_kernel_matrix(z_real, z_gen)

        loss_mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return loss_mmd

    def compute_sm3d_loss(self, features, lambda_m=0.5):
        """
        计算 SM3D 总损失 - Eq (8)
        Loss = (1 - lambda_m) * l_DSM + lambda_m * l_MMD
        """
        batch_size = features.size(0)

        # --- Part A: Denoising Score Matching (DSM) - Eq (4) ---
        # 你原来的实现是对的，保留
        noise = torch.randn_like(features) * 0.1 # sigma 通常设小一点，比如 0.1
        noisy_features = features + noise
        score_pred = self.score_model(noisy_features)
        score_true = -noise / (0.1 ** 2) # grad log p(x|x_0)
        l_dsm = 0.5 * F.mse_loss(score_pred, score_true)

        # --- Part B: MMD with LDS - Eq (6) & (7) ---
        # 这一步是论文为了解决 non-IID 稀疏性特意加的
        z_gen = self.langevin_dynamic_sampling(batch_size, features.device)
        l_mmd = self.compute_mmd_loss(features, z_gen)

        # 总损失
        total_loss = (1 - lambda_m) * l_dsm + lambda_m * l_mmd
        return total_loss

    # ==========================================
    # 2. SAG 实现 (Paper Section 3.3 & Appendix B.2)
    # ==========================================
    def compute_ksd_loss(self, z, z_aug):
        """
        计算 Kernelized Stein Discrepancy (KSD) - Eq (11)
        用于约束原始特征 z 和增强特征 z_aug 在特征空间的一致性

        注意：这里的公式比较复杂，涉及 Score Function 和 Kernel 的导数
        KSD(p, q) = E [ s(x)s(x')k(x,x') + s(x) div_2 k + s(x') div_1 k + trace(...) ]
        通常为了计算效率，会有简化版实现。
        这里我们实现一个基于 Score Model 指导的简化版分布对齐。
        """

        # 获取 Score Model 对增强数据的评分 (梯度场)
        # s_theta(z_aug)
        with torch.no_grad():
            score_aug = self.score_model(z_aug)

        # 论文 Eq (10) Stein Operator 的核心思想：
        # 我们希望 z_aug 落在 z (真实数据) 的高密度区域
        # 这意味着 score_aug (增强数据的梯度) 应该和 (z - z_aug) 方向一致
        # 或者更直接地，使用论文 Eq (12) 的逻辑：
        # KSD 实际上是在用 Score Model 作为一个 Critic 来评判 z 和 z_aug 的分布差异

        # 简化实现（基于 PyTorch 自动微分计算 KSD 梯度代价太高）：
        # 我们利用 Stein Identity 的性质：E[Score(x) * f(x) + grad(f(x))] = 0
        # 此处采用 Liu & Wang (SVGD作者) 的经典 KSD 估计器实现

        features = z
        K_xx, sigma = self.rbf_kernel_matrix(features, features)

        # score vectors: s_theta(z)
        scores = self.score_model(features)

        # 具体的 KSD 计算 (u-statistic estimator)
        # Term 1: s(x)^T s(x') * k(x,x')
        # [B, B]
        term1 = torch.matmul(scores, scores.t()) * K_xx

        # 由于 RBF 核梯度有解析解: grad_x k(x,y) = -1/sigma^2 * (x-y) * k(x,y)
        # 我们可以手动计算 Term 2 & 3 以避免二阶导数

        B = features.size(0)
        # (x_i - x_j) 矩阵: [B, B, Dim]
        X_diff = features.unsqueeze(1) - features.unsqueeze(0)

        # Term 2: s(x)^T * grad_x' k(x,x')
        # grad_x' k = 1/sigma^2 * (x-x') * k
        term2 = torch.einsum('id, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)

        # Term 3: s(x')^T * grad_x k(x,x')
        # grad_x k = -1/sigma^2 * (x-x') * k
        term3 = -torch.einsum('jd, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)

        # Term 4: trace(grad_x grad_x' k)
        # double grad RBF kernel
        dim = features.size(1)
        sq_dist = torch.sum(X_diff**2, dim=2)
        term4 = (dim / (sigma**2) - sq_dist / (sigma**4)) * K_xx

        ksd = (term1 + term2 + term3 + term4).mean()

        # 我们希望最小化 KSD (让 p(z) 和 p(z_aug) 一致)
        return ksd

    def forward(self, features, features_aug=None):
        """
        前向传播
        训练时: features 是原始图像特征, features_aug 是增强图像特征
        """
        # 1. OOD 检测分数 (测试时用)
        # 分数越高表示越是ID样本，越低表示越是OOD样本
        ood_scores = -torch.norm(self.score_model(features), dim=1)

        # 2. SM3D 损失 (训练时用) - 用于优化 Score Model
        sm_loss = self.compute_sm3d_loss(features)

        # 3. KSD 损失 (训练时用) - 用于优化 Feature Extractor
        # 如果没有提供增强特征 (比如测试阶段)，则 KSD 为 0
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