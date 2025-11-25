#!/usr/bin/env python3
"""
联邦学习客户端模块
基于项目工作文档中的FedRoD训练逻辑

作者: Claude Code
日期: 2025-11-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision.transforms as transforms
# AMP 功能已集成到 torch.amp 中，无需单独导入


class FLClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, foogd_module, train_loader, device):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            model: 本地模型
            foogd_module: FOOGD模块
            train_loader: 训练数据加载器
            device: 训练设备
        """
        self.client_id = client_id
        self.model = model
        self.foogd_module = foogd_module
        self.train_loader = train_loader
        self.device = device

        # 优化器
        # [删除或注释掉] 原来的 Adam
        # self.optimizer = torch.optim.Adam(...)

        # [修复] 确保包含 FOOGD 参数
        params = list(self.model.parameters())
        if self.foogd_module:
            params += list(self.foogd_module.parameters())

        # [新增] 使用 SGD，这是 FedAvg/FedRoD 的标配
        self.optimizer = torch.optim.SGD(
            params,
            lr=0.01,          # SGD 需要更大的学习率，Adam是1e-4，SGD建议 0.01 或 0.005
            momentum=0.9,     # 加上动量
            weight_decay=1e-5
        )

        # 损失权重
        self.lambda_ksd = 0.00001  # KSD损失权重
        self.lambda_sm = 0.005   # 评分匹配损失权重

        # 傅里叶增强参数
        self.use_fourier_aug = True  # 是否使用傅里叶增强
        self.fourier_beta = 0.15     # 幅度谱混合比例
        self.fourier_prob = 0.5      # 使用傅里叶增强的概率

        # 反归一化参数 (用于傅里叶增强)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        # [新增] 初始化 GradScaler (PyTorch 2.0+ API)
        self.scaler = torch.amp.GradScaler('cuda')

    def _fourier_augmentation(self, images, beta=None):
        """
        傅里叶数据增强 (向量化版本) - 极大提升速度
        直接对 [B, C, H, W] 整个批次进行运算，去除 for 循环
        """
        if beta is None:
            beta = self.fourier_beta

        # 1. 反归一化
        mean = self.mean.to(images.device).view(1, 3, 1, 1)
        std = self.std.to(images.device).view(1, 3, 1, 1)
        x_unnorm = images * std + mean

        # 2. 随机选择目标 (Shuffle Batch)
        batch_size = images.size(0)
        perm = torch.randperm(batch_size).to(images.device)
        target_x_unnorm = x_unnorm[perm] # 整个batch乱序作为target

        # 3. FFT 变换 (直接对整个 Batch 操作)
        # dim=(-2, -1) 表示只对最后两个维度(H, W)做变换，B和C维度自动保留
        fft_x = torch.fft.fftn(x_unnorm, dim=(-2, -1))
        fft_target = torch.fft.fftn(target_x_unnorm, dim=(-2, -1))

        # 4. 提取幅度谱和相位谱
        amp_x, pha_x = torch.abs(fft_x), torch.angle(fft_x)
        amp_target = torch.abs(fft_target)

        # 5. 混合幅度谱 (向量化操作)
        amp_new = (1.0 - beta) * amp_x + beta * amp_target

        # 6. 重建并逆变换
        fft_new = amp_new * torch.exp(1j * pha_x)
        x_aug_unnorm = torch.fft.ifftn(fft_new, dim=(-2, -1)).real

        # 7. 截断与重新归一化
        x_aug_unnorm = torch.clamp(x_aug_unnorm, 0, 1)
        x_aug = (x_aug_unnorm - mean) / std

        return x_aug

    def _apply_hybrid_augmentation(self, images):
        """
        应用混合增强
        增加概率控制
        """
        # 1. [关键修正] 概率控制
        # 只有当随机数小于阈值时才应用增强
        if torch.rand(1).item() < self.fourier_prob:
            # 你可以加上这一行来引入 beta 的随机性 (可选)
            # random_beta = np.random.uniform(0, self.fourier_beta * 2)
            return self._fourier_augmentation(images, beta=self.fourier_beta)
        else:
            # 否则返回原图
            return images

    def train_step(self, local_epochs=1):
        # [修复] 删除这里的 self.optimizer = ... 代码
        # 这一行必须删除！

        self.model.train()
        if self.foogd_module:
            self.foogd_module.train()

        total_loss = 0.0
        total_samples = 0

        # 用于记录分项 Loss
        epoch_log = {'cls': 0.0, 'ksd': 0.0, 'sm': 0.0}

        for epoch in range(local_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                # [修改] 使用 autocast 上下文管理器 (PyTorch 2.0+ API)
                with torch.amp.autocast('cuda'):
                    # 1. 数据增强 (仅使用傅里叶增强)
                    data_aug = self._apply_hybrid_augmentation(data)

                    # 2. 前向传播
                    logits_g, logits_p, features = self.model(data)
                    # 提示：如果你的显存足够，这里也可以计算增强数据的 logits
                    _, _, features_aug = self.model(data_aug)

                    # 3. 特征归一化 (防止 KSD 爆炸的关键)
                    features_norm = F.normalize(features, p=2, dim=1)
                    features_aug_norm = F.normalize(features_aug, p=2, dim=1)

                    # 4. 计算分类损失
                    loss_g = F.cross_entropy(logits_g, targets)
                    loss_p = F.cross_entropy(logits_p, targets)
                    classification_loss = loss_g + loss_p

                    # 5. 计算 FOOGD 损失 (所有计算都在 autocast 下进行)
                    foogd_loss = torch.tensor(0.0).to(self.device)
                    ksd_loss_val = 0.0
                    sm_loss_val = 0.0

                    if self.foogd_module:
                        # 注意：features 需要是 FP32 还是 FP16 取决于实现，通常 autocast 会自动处理
                        # 但如果出现数值不稳定，可以在这里暂时退出 autocast
                        ksd_loss, sm_loss, _ = self.foogd_module(features_norm, features_aug_norm)
                        # [关键调整] 降低权重，防止主导训练
                        # 建议先设得很小，跑通分类再说
                        foogd_loss = self.lambda_ksd * ksd_loss + self.lambda_sm * sm_loss

                        ksd_loss_val = ksd_loss.item()
                        sm_loss_val = sm_loss.item()

                    # 总损失
                    total_batch_loss = classification_loss + foogd_loss

                # [修改] 使用 scaler 进行反向传播和步进
                self.scaler.scale(total_batch_loss).backward()

                # 梯度裁剪 (Unscale 之后才能裁剪)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                if self.foogd_module:
                    torch.nn.utils.clip_grad_norm_(self.foogd_module.parameters(), max_norm=5.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 记录数据
                batch_size = data.size(0)
                total_loss += total_batch_loss.item() * batch_size
                total_samples += batch_size
                
                epoch_log['cls'] += classification_loss.item() * batch_size
                epoch_log['ksd'] += ksd_loss_val * batch_size
                epoch_log['sm'] += sm_loss_val * batch_size

        # 打印分项损失信息 (只打印第一个 epoch 的平均值)
        if total_samples > 0:
            print(f"  Cls: {epoch_log['cls']/total_samples:.4f} | "
                  f"KSD: {epoch_log['ksd']/total_samples:.4f} | "
                  f"SM: {epoch_log['sm']/total_samples:.4f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        generic_params = self.get_generic_parameters()

        return generic_params, avg_loss

    def get_generic_parameters(self):
        """获取通用参数 - 修复版"""
        # 直接返回整个模型的 state_dict (包含 BN stats)
        # 注意：FedRoD 理论上只聚合 Generic Head，但 backbone 的 BN 必须聚合
        # 简单起见，我们可以返回整个 backbone + head_g

        # 提取 backbone 和 head_g
        params = {}
        full_state = self.model.state_dict()

        for key, value in full_state.items():
            if 'head_p' not in key: # 排除个性化头
                params[key] = value.clone()

        return params

    def set_generic_parameters(self, generic_params):
        """设置通用参数 - 修复版"""
        # 使用 load_state_dict (strict=False 允许忽略 head_p)
        self.model.load_state_dict(generic_params, strict=False)

    def evaluate(self, test_loader):
        """
        在本地数据上评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            accuracy: 准确率
            loss: 平均损失
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                logits_g, logits_p, _ = self.model(data)
                loss_g = F.cross_entropy(logits_g, targets)
                loss_p = F.cross_entropy(logits_p, targets)
                loss = (loss_g + loss_p) / 2

                total_loss += loss.item() * data.size(0)

                # 使用通用头进行预测
                _, predicted = torch.max(logits_g, 1)
                correct += (predicted == targets).sum().item()
                total_samples += data.size(0)

        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return accuracy, avg_loss

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
                    _, _, ood_scores = self.foogd_module(features)
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
    accuracy, eval_loss = client.evaluate(client_loaders[0])
    print(f"评估准确率: {accuracy:.4f}")
    print(f"评估损失: {eval_loss:.4f}")

    print("客户端测试完成!")