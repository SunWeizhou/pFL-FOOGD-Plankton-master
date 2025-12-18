#!/usr/bin/env python3
"""
联邦学习客户端模块 (修正版)
包含: Sigmoid Warm-up, LR=0.001, 显存优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import collections
import numpy as np  # 必须导入 numpy
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

from models import TaxonomyLoss

class FLClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, foogd_module, train_loader, device,
                 compute_aug_features=True, freeze_bn=True, base_lr=0.001,
                 algorithm='fedrod', use_taxonomy=False, taxonomy_matrix=None):  # <--- 新增参数
        self.client_id = client_id
        self.model = model
        self.foogd_module = foogd_module
        self.train_loader = train_loader
        self.device = device
        self.compute_aug_features = compute_aug_features
        self.freeze_bn = freeze_bn
        self.base_lr = base_lr  # 基础学习率，可根据 batch_size 调整
        self.algorithm = algorithm  # 算法选择: 'fedavg' 或 'fedrod'
        self.use_taxonomy = use_taxonomy  # <--- 保存开关状态

        # 2. 根据开关初始化 Loss 函数
        # 只要开了开关，就初始化 Taxonomy Loss
        if self.use_taxonomy:  # 去掉算法限制，只要开了开关就初始化
            if taxonomy_matrix is None:
                # 如果未传入矩阵，则尝试构建（向后兼容）
                try:
                    from data_utils import build_taxonomy_matrix
                    taxonomy_matrix = build_taxonomy_matrix(num_classes=54, device=self.device)
                    print(f"Client {self.client_id}: 自动构建 Taxonomy 矩阵")
                except Exception as e:
                    print(f"Client {self.client_id}: 警告 - 无法构建 Taxonomy 矩阵: {e}")
                    self.tax_loss_fn = None
                    return
            print(f"Client {self.client_id}: Taxonomy-Aware Loss 已启用 ✅ (Alg: {self.algorithm})")
            # 使用传入的矩阵
            self.tax_loss_fn = TaxonomyLoss(taxonomy_matrix=taxonomy_matrix, lambda_t=0.5)
        else:
            self.tax_loss_fn = None

        # 1. 在初始化时定义优化器 (只做一次)
        self.optimizer_main = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,  # 使用传入的基础学习率
            momentum=0.9,
            weight_decay=1e-5
        )

        # FOOGD 优化器
        if self.foogd_module:
            self.optimizer_foogd = torch.optim.Adam(
                self.foogd_module.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999)
            )

        self.lambda_ksd = 0.01
        self.lambda_sm = 0.1

        # 傅里叶增强参数
        self.use_fourier_aug = True
        self.fourier_beta = 0.4
        self.fourier_prob = 0.9
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda')

    def _fourier_augmentation(self, images, beta=None):
        """向量化傅里叶增强"""
        if beta is None: beta = self.fourier_beta
        mean = self.mean.to(images.device).view(1, 3, 1, 1)
        std = self.std.to(images.device).view(1, 3, 1, 1)
        x_unnorm = images * std + mean

        batch_size = images.size(0)
        perm = torch.randperm(batch_size).to(images.device)
        target_x_unnorm = x_unnorm[perm]

        fft_x = torch.fft.fftn(x_unnorm, dim=(-2, -1))
        fft_target = torch.fft.fftn(target_x_unnorm, dim=(-2, -1))

        amp_x, pha_x = torch.abs(fft_x), torch.angle(fft_x)
        amp_target = torch.abs(fft_target)

        amp_new = (1.0 - beta) * amp_x + beta * amp_target
        fft_new = amp_new * torch.exp(1j * pha_x)

        x_aug_unnorm = torch.fft.ifftn(fft_new, dim=(-2, -1)).real
        x_aug_unnorm = torch.clamp(x_aug_unnorm, 0, 1)
        x_aug = (x_aug_unnorm - mean) / std
        return x_aug

    def _apply_hybrid_augmentation(self, images):
        if torch.rand(1).item() < self.fourier_prob:
            return self._fourier_augmentation(images, beta=self.fourier_beta)
        return images

    # [修正2] 接收 current_round 参数并实现 Sigmoid Warm-up
    def train_step(self, local_epochs=1, current_round=0):
        # =================================================================
        # 【优化】: 不再每次重新创建优化器，只调整学习率
        # 这样可以保留优化器的状态（如 momentum），同时确保学习率正确
        # =================================================================
        # 3. 动态调整学习率 (可选，但推荐)
        # 这是一个小技巧：虽然不重置优化器，但我们要确保学习率是正确的
        # 如果你有 decay 逻辑，可以在这里重新赋值 lr
        target_lr = self.base_lr  # 使用基础学习率
        # 建议：如果 Batch=64, 这里可以尝试 0.01 或保持 0.001
        for param_group in self.optimizer_main.param_groups:
            param_group['lr'] = target_lr

        # 如果有FOOGD模块，也调整其学习率
        if self.foogd_module:
            for param_group in self.optimizer_foogd.param_groups:
                param_group['lr'] = 1e-3

        self.model.train()
        if self.foogd_module:
            self.foogd_module.train()

        # [修复] 强制冻结 BN 层 (如果 freeze_bn=True)
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        total_loss = 0.0
        total_samples = 0
        epoch_log = {'cls': 0.0, 'ksd': 0.0, 'sm': 0.0}

        # --- 智能动态权重 (Sigmoid Warm-up) ---
        if self.foogd_module:
            warm_up_center = 10
            slope = 0.5
            # 计算 alpha (0~1)
            alpha = 1 / (1 + np.exp(-slope * (current_round - warm_up_center)))

            target_lambda_ksd = 0.01
            target_lambda_sm = 0.1

            effective_lambda_ksd = target_lambda_ksd * alpha
            effective_lambda_sm = target_lambda_sm * alpha

            # 打印当前权重
            if current_round % 5 == 0 and current_round > 0:
                print(f"  [Auto-Weight] Round {current_round}: Alpha={alpha:.4f} | KSD_w={effective_lambda_ksd:.6f}")
        else:
            effective_lambda_ksd = 0.0
            effective_lambda_sm = 0.0
        # -----------------------------------

        for epoch in range(local_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer_main.zero_grad()
                if self.foogd_module:
                    self.optimizer_foogd.zero_grad()

                with torch.amp.autocast('cuda'):
                    data_aug = self._apply_hybrid_augmentation(data)
                    logits_g, logits_p, features = self.model(data)
                    _, _, features_aug = self.model(data_aug)

                    # 归一化 (关键)
                    features_norm = F.normalize(features, p=2, dim=1)
                    features_aug_norm = F.normalize(features_aug, p=2, dim=1)

                    # =================================================
                    # 3. 这里的 Loss 计算逻辑要改成带开关的
                    # =================================================

                    # --- 计算 Head-P 的 Loss (永远是 CrossEntropy) ---
                    loss_p = F.cross_entropy(logits_p, targets)

                    # --- 统一计算 Head-G 的 Loss ---
                    if self.use_taxonomy and self.tax_loss_fn is not None:
                        loss_g = self.tax_loss_fn(logits_g, targets)
                    else:
                        loss_g = F.cross_entropy(logits_g, targets)

                    # --- 根据算法组合总 Loss ---
                    if self.algorithm == 'fedavg':
                        classification_loss = loss_g
                    else: # FedRoD
                        classification_loss = loss_g + loss_p

                    # [修复] 初始化所有损失变量
                    ksd_loss = torch.tensor(0.0, device=self.device)
                    ksd_loss_val = 0.0
                    sm_loss = torch.tensor(0.0, device=self.device)
                    sm_loss_val = 0.0
                    loss_for_foogd = torch.tensor(0.0, device=self.device)

                    if self.foogd_module:
                        ksd_loss, sm_loss, _ = self.foogd_module(features_norm, features_aug_norm)
                        ksd_loss_val = ksd_loss.item()
                        sm_loss_val = sm_loss.item()
                        loss_for_foogd = sm_loss

                    # 应用动态权重
                    loss_for_main = classification_loss + effective_lambda_ksd * ksd_loss

                # [修正3] 删除 retain_graph=True，释放显存
                self.scaler.scale(loss_for_main).backward()
                self.scaler.unscale_(self.optimizer_main)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer_main)

                if self.foogd_module:
                    self.scaler.scale(loss_for_foogd).backward()
                    self.scaler.unscale_(self.optimizer_foogd)
                    torch.nn.utils.clip_grad_norm_(self.foogd_module.parameters(), 5.0)
                    self.scaler.step(self.optimizer_foogd)

                self.scaler.update()

                batch_size = data.size(0)
                # [修复] 统一使用标量值进行统计计算
                total_batch_loss = classification_loss.item() + effective_lambda_ksd * ksd_loss_val
                total_loss += total_batch_loss * batch_size
                total_samples += batch_size
                epoch_log['cls'] += classification_loss.item() * batch_size
                epoch_log['ksd'] += ksd_loss_val * batch_size
                epoch_log['sm'] += sm_loss_val * batch_size

        if total_samples > 0:
            # 计算平均损失并打印
            avg_cls = epoch_log['cls'] / total_samples
            avg_ksd = epoch_log['ksd'] / total_samples
            avg_sm = epoch_log['sm'] / total_samples
            print(f"Client {self.client_id} - Epochs {local_epochs} - Avg Loss: {total_loss / total_samples:.4f} | "
                  f"Cls: {avg_cls:.4f}, KSD: {avg_ksd:.6f}, SM: {avg_sm:.6f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        generic_params = self.get_generic_parameters()
        return generic_params, avg_loss

    def get_generic_parameters(self):
        params = {}
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            if 'head_p' not in key:
                params[f"model.{key}"] = value.clone()
        if self.foogd_module:
            foogd_state = self.foogd_module.state_dict()
            for key, value in foogd_state.items():
                params[f"foogd.{key}"] = value.clone()
        return params

    def set_generic_parameters(self, generic_params):
        model_params = {}
        foogd_params = {}
        for key, value in generic_params.items():
            if key.startswith("model."):
                model_params[key.replace("model.", "")] = value
            elif key.startswith("foogd."):
                foogd_params[key.replace("foogd.", "")] = value
        self.model.load_state_dict(model_params, strict=False)
        if self.foogd_module and foogd_params:
            self.foogd_module.load_state_dict(foogd_params, strict=False)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        correct_g = 0
        correct_p = 0
        total_samples = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits_g, logits_p, _ = self.model(data)
                loss_g = F.cross_entropy(logits_g, targets)
                loss_p = F.cross_entropy(logits_p, targets)
                loss = (loss_g + loss_p) / 2
                total_loss += loss.item() * data.size(0)
                _, pred_g = torch.max(logits_g, 1)
                correct_g += (pred_g == targets).sum().item()
                _, pred_p = torch.max(logits_p, 1)
                correct_p += (pred_p == targets).sum().item()
                total_samples += data.size(0)
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'acc_g': correct_g / total_samples if total_samples > 0 else 0.0,
            'acc_p': correct_p / total_samples if total_samples > 0 else 0.0
        }

    def compute_ood_scores(self, data_loader):
        """
        计算OOD分数

        Args:
            data_loader: 数据加载器

        Returns:
            ood_scores: OOD分数列表
            labels: 真实标签列表
        """
        self.model.eval()
        if self.foogd_module:
            self.foogd_module.eval()

        all_ood_scores = []
        all_labels = []

        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                _, _, features = self.model(data)

                if self.foogd_module:
                    # [修复] 必须先归一化，与训练时保持一致！
                    features_norm = F.normalize(features, p=2, dim=1)
                    _, _, ood_scores = self.foogd_module(features_norm)
                else:
                    # 如果没有FOOGD模块，使用特征范数作为OOD分数
                    ood_scores = torch.norm(features, dim=1)

                all_ood_scores.extend(ood_scores.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        return all_ood_scores, all_labels

    def evaluate_comprehensive(self, id_loader, inc_loader, near_ood_loader, far_ood_loader):
        """
        [新增] 全面评估 Head-P 的性能 (ID, IN-C, Near, Far)
        """
        self.model.eval()
        if self.foogd_module:
            self.foogd_module.eval()

        metrics = {}

        # 1. 评估 ID Accuracy (Head-P)
        # -------------------------------------------------
        correct_p = 0
        total_id = 0
        id_scores = []  # 缓存 ID 数据的 OOD 分数，用于后续计算 AUROC

        with torch.no_grad():
            for data, targets in id_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits_g, logits_p, features = self.model(data)

                # 计算 Acc
                _, pred_p = torch.max(logits_p, 1)
                correct_p += (pred_p == targets).sum().item()
                total_id += data.size(0)

                # 计算 OOD Score (用于后续对比)
                if self.foogd_module:
                    features_norm = F.normalize(features, p=2, dim=1)
                    _, _, scores = self.foogd_module(features_norm)
                else:
                    scores = torch.norm(features, dim=1)
                id_scores.extend(scores.cpu().numpy())

        metrics['acc_p_id'] = correct_p / total_id if total_id > 0 else 0.0

        # 2. 评估 IN-C Accuracy (Head-P)
        # -------------------------------------------------
        # 注意：这里我们简化处理，直接用全量 IN-C 测试集。
        # 严谨做法是像 ID 一样只测客户端见过的类别，但全量测更能反映真实泛化能力。
        if inc_loader:
            correct_p_inc = 0
            total_inc = 0
            with torch.no_grad():
                for data, targets in inc_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    _, logits_p, _ = self.model(data)

                    # 只统计 ID 范围内的类别 (0-53)
                    valid_mask = targets >= 0
                    if valid_mask.any():
                        v_data = data[valid_mask]
                        v_targets = targets[valid_mask]
                        v_logits = logits_p[valid_mask]

                        _, pred_p = torch.max(v_logits, 1)
                        correct_p_inc += (pred_p == v_targets).sum().item()
                        total_inc += v_targets.size(0)

            metrics['acc_p_inc'] = correct_p_inc / total_inc if total_inc > 0 else 0.0

        # 3. 评估 OOD Detection (AUROC)
        # -------------------------------------------------
        # 辅助函数
        def get_scores(loader):
            s_list = []
            with torch.no_grad():
                for data, _ in loader:
                    data = data.to(self.device)
                    _, _, features = self.model(data)
                    if self.foogd_module:
                        features_norm = F.normalize(features, p=2, dim=1)
                        _, _, s = self.foogd_module(features_norm)
                    else:
                        s = torch.norm(features, dim=1)
                    s_list.extend(s.cpu().numpy())
            return np.array(s_list)

        id_scores = np.array(id_scores)

        # Near-OOD
        if near_ood_loader:
            near_scores = get_scores(near_ood_loader)
            # 标签：ID=0, OOD=1 (AUROC计算通常假设正例分数高，这里 FOOGD 分数 ID 高，所以反过来或者用 1-score)
            # 修正：sklearn roc_auc_score: y_true, y_score.
            # 如果 ID score 高 (norm大), OOD score 低.
            # 设 ID=1, OOD=0, 则 AUROC 反映 score 区分度.
            y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(near_scores)])
            y_scores = np.concatenate([id_scores, near_scores])
            metrics['auroc_p_near'] = roc_auc_score(y_true, y_scores)

        # Far-OOD
        if far_ood_loader:
            far_scores = get_scores(far_ood_loader)
            y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(far_scores)])
            y_scores = np.concatenate([id_scores, far_scores])
            metrics['auroc_p_far'] = roc_auc_score(y_true, y_scores)

        return metrics


if __name__ == "__main__":
    # 测试客户端
    print("测试联邦学习客户端...")

    # 创建模型和数据加载器
    from models import create_fedrod_model
    from data_utils import create_federated_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据加载器
    data_root = "./data"
    client_loaders, _, _, _ = create_federated_loaders(
        data_root, n_clients=3, batch_size=4, image_size=224
    )

    # 创建客户端
    model, foogd_module = create_fedrod_model()
    model = model.to(device)
    if foogd_module:
        foogd_module = foogd_module.to(device)

    client = FLClient(
        client_id=0,
        model=model,
        foogd_module=foogd_module,
        train_loader=client_loaders[0],
        device=device
    )

    # 测试训练步骤
    print("\n测试客户端训练...")
    generic_params, train_loss = client.train_step(local_epochs=1)
    print(f"训练损失: {train_loss:.4f}")
    print(f"通用参数数量: {len(generic_params)}")

    # 测试评估
    print("\n测试客户端评估...")
    eval_metrics = client.evaluate(client_loaders[0])
    print(f"通用头准确率: {eval_metrics['acc_g']:.4f}")
    print(f"个性化头准确率: {eval_metrics['acc_p']:.4f}")
    print(f"评估损失: {eval_metrics['loss']:.4f}")

    print("客户端测试完成!")