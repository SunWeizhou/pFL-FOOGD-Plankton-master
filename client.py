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
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + (list(self.foogd_module.parameters()) if foogd_module else []),
            lr=1e-4, weight_decay=1e-5
        )

        # 损失权重
        self.lambda_ksd = 0.01  # KSD损失权重
        self.lambda_sm = 0.01   # 评分匹配损失权重

        # 数据增强变换
        self._setup_augmentations()

    def _setup_augmentations(self):
        """设置数据增强变换"""
        # 强增强：用于SAG的增强特征
        self.strong_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

    def _apply_strong_augmentation(self, images):
        """
        应用强数据增强

        Args:
            images: 原始图像张量 [batch_size, 3, height, width]

        Returns:
            augmented_images: 增强后的图像张量
        """
        import torchvision.transforms as transforms
        batch_size = images.size(0)
        augmented_images = []

        for i in range(batch_size):
            # 将张量转换回PIL图像进行增强
            img = transforms.ToPILImage()(images[i])
            # 应用强增强
            img_aug = self.strong_augmentation(img)
            augmented_images.append(img_aug)

        return torch.stack(augmented_images).to(images.device)

     def train_step(self, local_epochs=1):
        self.model.train()
        if self.foogd_module:
            self.foogd_module.train()

        total_loss = 0.0
        total_samples = 0
        
        # [Debug] 用于记录分项 Loss，看看是谁在捣乱
        epoch_log = {'cls': 0.0, 'ksd': 0.0, 'sm': 0.0}

        for epoch in range(local_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                # 1. 数据增强
                data_aug = self._apply_strong_augmentation(data)

                # 2. 前向传播
                logits_g, logits_p, features = self.model(data)
                _, _, features_aug = self.model(data_aug)

                # 3. 特征归一化 (防止 KSD 爆炸的关键)
                features_norm = F.normalize(features, p=2, dim=1)
                features_aug_norm = F.normalize(features_aug, p=2, dim=1)

                # 4. 计算分类损失
                loss_g = F.cross_entropy(logits_g, targets)
                loss_p = F.cross_entropy(logits_p, targets)
                classification_loss = loss_g + loss_p

                # 5. 计算 FOOGD 损失
                foogd_loss = torch.tensor(0.0).to(self.device)
                ksd_loss_val = 0.0
                sm_loss_val = 0.0
                
                if self.foogd_module:
                    ksd_loss, sm_loss, _ = self.foogd_module(features_norm, features_aug_norm)
                    # [关键调整] 降低权重，防止主导训练
                    # 建议先设得很小，跑通分类再说
                    foogd_loss = self.lambda_ksd * ksd_loss + self.lambda_sm * sm_loss
                    
                    ksd_loss_val = ksd_loss.item()
                    sm_loss_val = sm_loss.item()

                # 总损失
                total_batch_loss = classification_loss + foogd_loss

                # 6. 反向传播与梯度裁剪
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                
                # [新增] 梯度裁剪：防止 NaN 的核武器
                # max_norm 通常设为 1.0 或 5.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                if self.foogd_module:
                    torch.nn.utils.clip_grad_norm_(self.foogd_module.parameters(), max_norm=5.0)
                
                self.optimizer.step()

                # 记录数据
                batch_size = data.size(0)
                total_loss += total_batch_loss.item() * batch_size
                total_samples += batch_size
                
                epoch_log['cls'] += classification_loss.item() * batch_size
                epoch_log['ksd'] += ksd_loss_val * batch_size
                epoch_log['sm'] += sm_loss_val * batch_size

        # 打印调试信息 (只打印第一个 epoch 的平均值)
        if total_samples > 0:
            print(f"  [Debug] Cls: {epoch_log['cls']/total_samples:.4f} | "
                  f"KSD: {epoch_log['ksd']/total_samples:.4f} | "
                  f"SM: {epoch_log['sm']/total_samples:.4f}")

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        generic_params = self.get_generic_parameters()

        return generic_params, avg_loss

    def get_generic_parameters(self):
        """
        获取通用参数（服务器聚合的部分）

        Returns:
            generic_params: 通用参数状态字典
        """
        generic_params = {}

        # 获取骨干网络和通用头的参数
        for name, param in self.model.named_parameters():
            if 'head_p' not in name:  # 排除个性化头
                generic_params[name] = param.data.clone()

        return generic_params

    def set_generic_parameters(self, generic_params):
        """
        设置通用参数（从服务器接收）

        Args:
            generic_params: 通用参数状态字典
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'head_p' not in name and name in generic_params:
                    param.data.copy_(generic_params[name])

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