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

        # 1. 主模型优化器 (Backbone + Classifiers) -> 使用 SGD
        self.optimizer_main = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-5
        )

        # 2. FOOGD 优化器 (Score Model) -> 使用 Adam
        if self.foogd_module:
            self.optimizer_foogd = torch.optim.Adam(
                self.foogd_module.parameters(),
                lr=1e-3,  # Adam 标准学习率
                betas=(0.9, 0.999)
            )

        # 损失权重
        self.lambda_ksd = 0.01  # KSD损失权重
        self.lambda_sm = 0.1   # 评分匹配损失权重

        # 傅里叶增强参数
        self.use_fourier_aug = True  # 是否使用傅里叶增强
        self.fourier_beta = 0.4     # 幅度谱混合比例
        self.fourier_prob = 0.9      # 使用傅里叶增强的概率

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

                # 清零两个优化器的梯度
                self.optimizer_main.zero_grad()
                if self.foogd_module:
                    self.optimizer_foogd.zero_grad()

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
                    ksd_loss_val = 0.0
                    sm_loss_val = 0.0

                    if self.foogd_module:
                        # 注意：features 需要是 FP32 还是 FP16 取决于实现，通常 autocast 会自动处理
                        # 但如果出现数值不稳定，可以在这里暂时退出 autocast
                        ksd_loss, sm_loss, _ = self.foogd_module(features_norm, features_aug_norm)

                        ksd_loss_val = ksd_loss.item()
                        sm_loss_val = sm_loss.item()

                    # 关键修改：Loss 组合
                    # 1. 用于更新主模型的 Loss (分类 + KSD)
                    loss_for_main = classification_loss + self.lambda_ksd * ksd_loss

                    # 2. 用于更新 FOOGD 的 Loss (SM Loss)
                    # 注意：SM Loss 只用于更新 ScoreNet，不应该回传给 Backbone
                    # (我们在 compute_sm3d_loss 里已经 detach 了 features，所以这里直接用即可)
                    loss_for_foogd = sm_loss

                # 反向传播与更新
                # 注意：由于用了 scaler，需要分别处理

                # Update Main Model
                self.scaler.scale(loss_for_main).backward() # retain_graph=True 如果需要共用计算图
                self.scaler.unscale_(self.optimizer_main)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer_main)

                # Update FOOGD Module
                if self.foogd_module:
                    self.scaler.scale(loss_for_foogd).backward()
                    self.scaler.unscale_(self.optimizer_foogd)
                    torch.nn.utils.clip_grad_norm_(self.foogd_module.parameters(), 5.0)
                    self.scaler.step(self.optimizer_foogd)

                self.scaler.update()

                # 记录数据
                batch_size = data.size(0)
                # 计算总损失用于记录 (分类 + KSD + SM)
                if self.foogd_module:
                    total_batch_loss = classification_loss + self.lambda_ksd * ksd_loss + self.lambda_sm * sm_loss
                else:
                    total_batch_loss = classification_loss
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
        params = {}

        # 1. 获取主模型参数 (排除 head_p)
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            if 'head_p' not in key:
                params[f"model.{key}"] = value.clone() # 添加前缀以区分

        # 2. 获取 FOOGD 参数 [新增]
        if self.foogd_module:
            foogd_state = self.foogd_module.state_dict()
            for key, value in foogd_state.items():
                params[f"foogd.{key}"] = value.clone() # 添加前缀

        return params

    def set_generic_parameters(self, generic_params):
        # 分离参数
        model_params = {}
        foogd_params = {}

        for key, value in generic_params.items():
            if key.startswith("model."):
                model_params[key.replace("model.", "")] = value
            elif key.startswith("foogd."):
                foogd_params[key.replace("foogd.", "")] = value

        # 加载参数
        self.model.load_state_dict(model_params, strict=False)
        if self.foogd_module and foogd_params:
            self.foogd_module.load_state_dict(foogd_params, strict=False)

    def evaluate(self, test_loader):
        """
        在测试集上评估模型
        返回:
            metrics: 包含 accuracy_g, accuracy_p, loss 的字典
        """
        self.model.eval()

        total_loss = 0.0
        correct_g = 0
        correct_p = 0  # [新增] 记录 head_p 的正确数
        total_samples = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                logits_g, logits_p, _ = self.model(data)

                # 计算 Loss (这里保持原样，或者你可以只返回分类 Loss)
                loss_g = F.cross_entropy(logits_g, targets)
                loss_p = F.cross_entropy(logits_p, targets)
                loss = (loss_g + loss_p) / 2
                total_loss += loss.item() * data.size(0)

                # [原有] 计算 head_g 准确率
                _, pred_g = torch.max(logits_g, 1)
                correct_g += (pred_g == targets).sum().item()

                # [新增] 计算 head_p 准确率
                _, pred_p = torch.max(logits_p, 1)
                correct_p += (pred_p == targets).sum().item()

                total_samples += data.size(0)

        # 计算平均指标
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        acc_g = correct_g / total_samples if total_samples > 0 else 0.0
        acc_p = correct_p / total_samples if total_samples > 0 else 0.0 # [新增]

        # 返回字典更清晰
        return {
            'loss': avg_loss,
            'acc_g': acc_g,
            'acc_p': acc_p
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