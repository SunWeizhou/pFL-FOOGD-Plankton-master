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

class FLClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, foogd_module, train_loader, device, compute_aug_features=True, freeze_bn=True):
        self.client_id = client_id
        self.model = model
        self.foogd_module = foogd_module
        self.train_loader = train_loader
        self.device = device
        self.compute_aug_features = compute_aug_features
        self.freeze_bn = freeze_bn

        # [修正1] 学习率改为 0.001 (适配 DenseNet)
        self.optimizer_main = torch.optim.SGD(
            self.model.parameters(),
            lr=0.001,
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
        # 【修正 1】: 每次接收全局模型后，必须重置优化器！
        # 否则上一轮的 Momentum 会作用在这一轮的新参数上，导致严重震荡。
        # =================================================================
        current_lr = self.optimizer_main.param_groups[0]['lr']  # 保持当前的 LR
        self.optimizer_main = torch.optim.SGD(
            self.model.parameters(),
            lr=current_lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        # 注意：如果你之前使用了 self.optimizer_main.state = collections.defaultdict(dict)
        # 请确保你在文件头部 import collections，否则会报错。
        # 重新实例化是最稳妥的方法。

        # 如果有FOOGD模块，也重置其优化器
        if self.foogd_module:
            self.optimizer_foogd = torch.optim.Adam(
                self.foogd_module.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999)
            )

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

                    loss_g = F.cross_entropy(logits_g, targets)
                    loss_p = F.cross_entropy(logits_p, targets)
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

            # 只在某些轮次打印详细损失
            if current_round % 5 == 0:
                print(f"  Client {self.client_id} Losses - CLS: {avg_cls:.4f}, KSD: {avg_ksd:.6f}, SM: {avg_sm:.6f}")

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