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

from data_utils import build_taxonomy_matrix


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

    def rbf_kernel_matrix(self, X, Y, sigma=None, sq_dist=None):
        """
        计算 RBF 核矩阵 K(X, Y)

        Args:
            X, Y: 输入特征
            sigma: 核带宽 (如果为 None 则使用中位数启发式)
            sq_dist: [可选] 预先计算好的距离平方矩阵 ||X-Y||^2
                     如果提供了这个，就不会重复计算 cdist
        """
        # 1. 如果没有提供预计算距离，则现场计算
        if sq_dist is None:
            # cdist 计算的是欧氏距离 ||x-y||
            dists = torch.cdist(X, Y)
            sq_dist = dists ** 2
        else:
            # 如果有了平方距离，开根号得到欧氏距离 (用于计算 sigma)
            dists = torch.sqrt(torch.clamp(sq_dist, min=1e-8))

        # 2. 计算 Sigma (中位数启发式)
        if sigma is None:
            sigma = torch.median(dists).detach()

        # 3. 计算核矩阵
        # K(x,y) = exp(- ||x-y||^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * sigma**2 + 1e-8)

        # 直接使用 sq_dist
        K_XY = torch.exp(-gamma * sq_dist)

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
        优化版：只计算一次距离矩阵
        """
        # 1. 准备数据
        features = z_aug
        scores = self.score_model(features).detach()
        batch_size = features.size(0)

        # 2. 【关键优化】先计算差分向量和距离平方
        # X_diff: (x_i - x_j) -> [B, B, Dim]
        # 显存优化提示：如果 batch_size 很大，这里可能会 OOM，可以考虑切片处理
        X_diff = features.unsqueeze(1) - features.unsqueeze(0)

        # sq_dist: ||x_i - x_j||^2 -> [B, B]
        sq_dist = torch.sum(X_diff**2, dim=2)

        # 3. 调用核函数 (传入预先计算的 sq_dist)
        K_xx, sigma = self.rbf_kernel_matrix(features, features, sq_dist=sq_dist)

        # --- 以下计算逻辑保持不变，但利用了已有的变量 ---

        # Term 1: s(x)^T s(x') * k(x,x')
        term1 = torch.matmul(scores, scores.t()) * K_xx

        # Term 2: s(x)^T * (x - x') ...
        # 注意：grad_x' k(x,x') = 1/sigma^2 * (x - x') * k
        # scores: [B, D], X_diff: [B, B, D] -> einsum -> [B, B]
        term2 = torch.einsum('id, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)

        # Term 3: s(x')^T * (x' - x) ... (注意正负号)
        term3 = -torch.einsum('jd, ijd -> ij', scores, X_diff) * K_xx / (sigma**2 + 1e-8)

        # Term 4: trace term
        dim = features.size(1)
        # 这里直接复用 sq_dist，不需要重新 sum(X_diff**2)
        term4 = (dim / (sigma**2) - sq_dist / (sigma**4)) * K_xx

        # 最终 Loss
        ksd = (term1 + term2 + term3 + term4).mean()

        # [修改] 归一化：除以特征维度，让 KSD 数值从 ~2000 降到 ~2.0
        return ksd / dim

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


class TaxonomyLoss(nn.Module):
    """
    层级感知损失函数 (Taxonomy-Aware Loss)
    Loss = CrossEntropy + lambda * Expected_Tree_Distance

    理论支撑：Tree-Regularized Loss 或 Cost-Sensitive Learning
    """
    def __init__(self, taxonomy_matrix, lambda_t=0.5):
        """
        初始化层级感知损失函数

        Args:
            taxonomy_matrix: 分类学代价矩阵 [num_classes, num_classes]
            lambda_t: 层级正则化项的权重系数
        """
        super(TaxonomyLoss, self).__init__()
        self.lambda_t = lambda_t
        # 直接使用传入的矩阵
        self.matrix = taxonomy_matrix
        self.num_classes = taxonomy_matrix.size(0) if hasattr(taxonomy_matrix, 'size') else taxonomy_matrix.shape[0]

    def forward(self, logits, targets):
        """
        前向传播计算损失

        Args:
            logits: 模型输出 [Batch, Classes]
            targets: 真实标签 [Batch]

        Returns:
            total_loss: 总损失 = 交叉熵损失 + lambda_t * 期望代价
        """
        # 1. 基础分类损失 (Cross Entropy)
        ce_loss = F.cross_entropy(logits, targets)

        # 2. 层级正则化项
        # 计算预测的概率分布 P(y_pred | x)
        probs = F.softmax(logits, dim=1)  # [B, 54]

        # 获取每个样本对应的真实类别的代价行向量
        # targets 维度 [B], matrix 维度 [54, 54]
        # selected_costs 维度 [B, 54] -> 第 i 行表示真实类别 targets[i] 到其他所有类别的距离
        selected_costs = self.matrix[targets]

        # 计算期望代价 (Expected Cost)
        # sum( P(j) * Distance(truth, j) )
        # 如果模型把高概率给了距离 truth 很远的类别，这个值会很大
        tree_reg = torch.sum(probs * selected_costs, dim=1).mean()

        # 3. 总损失
        total_loss = ce_loss + self.lambda_t * tree_reg

        return total_loss

    def get_expected_cost(self, logits, targets):
        """
        计算期望代价（不包含交叉熵损失）

        Args:
            logits: 模型输出 [Batch, Classes]
            targets: 真实标签 [Batch]

        Returns:
            expected_cost: 期望代价
        """
        probs = F.softmax(logits, dim=1)
        selected_costs = self.matrix[targets]
        expected_cost = torch.sum(probs * selected_costs, dim=1).mean()
        return expected_cost