#!/usr/bin/env python3
"""
数据工具模块 - 用于联邦学习的数据划分和加载
基于项目工作文档中的严格类别定义

作者: Claude Code
日期: 2025-11-22
"""

import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 严格类别定义 - 根据新项目工作文档
ID_CLASSES = [
    "Polychaeta_most with eggs", "Polychaeta Type A", "Polychaeta Type B", "Polychaeta Type C",
    "Polychaeta Type D", "Polychaeta Type E", "Polychaeta Type F", "Penilia avirostris",
    "Evadne tergestina", "Acartia sp.A", "Acartia sp.B", "Acartia sp.C", "Calanopia sp.",
    "Labidocera sp.", "Tortanus gracilis", "Calanoid with egg", "Calanoid Type A",
    "Calanoid Type B", "Oithona sp.B with egg", "Cyclopoid Type A with egg",
    "Harpacticoid mating", "Microsetella sp.", "Caligus sp.", "Copepod Type A", "Caprella sp.",
    "Amphipoda Type A", "Amphipoda Type B", "Amphipoda Type C", "Gammarids Type A",
    "Gammarids Type B", "Gammarids Type C", "Cymodoce sp.", "Lucifer sp.", "Macrura larvae",
    "Megalopa larva Phase 1 Type B", "Megalopa larva Phase 1 Type C",
    "Megalopa larva Phase 1 Type D", "Megalopa larva_Phase 2", "Porcrellanidae larva",
    "Shrimp-like larva Type A", "Shrimp-like larva Type B", "Shrimp-like Type A",
    "Shrimp-like Type B", "Shrimp-like Type D", "Shrimp-like Type F", "Cumacea Type A",
    "Cumacea Type B", "Chaetognatha", "Oikopleura sp. parts", "Tunicata Type A",
    "Jellyfish", "Creseis acicula", "Noctiluca scintillans", "Phaeocystis globosa"
]

NEAR_OOD_CLASSES = [
    "Polychaeta larva", "Calanoid Nauplii", "Calanoid Type C", "Calanoid Type D",
    "Oithona sp.A with egg", "Cyclopoid Type A", "Harpacticoid", "Monstrilla sp.A",
    "Monstrilla sp.B", "Megalopa larva Phase 1 Type A", "Shrimp-like Type C",
    "Shrimp-like Type E", "Ostracoda", "Oikopleura sp.", "Actiniaria larva", "Hydroid",
    "Jelly-like", "Bryozoan larva", "Gelatinous Zooplankton", "Unknown Type A",
    "Unknown Type B", "Unknown Type C", "Unknown Type D", "Balanomorpha exuviate",
    "Monstrilloid", "Fish Larvae"
]

FAR_OOD_CLASSES = [
    "Crustacean limb Type A", "Crustacean limb_Type B", "Fish egg",
    "Particle filamentous Type A", "Particle filamentous Type B", "Particle bluish",
    "Particle molts", "Particle translucent flocs", "Particle_yellowish flocs",
    "Particle_yellowish rods", "Bubbles", "Fish tail"
]


# data_utils.py

class PlanktonDataset(Dataset):
    """浮游生物数据集类 - 支持缓存加速"""

    def __init__(self, root_dir, transform=None, mode='train', client_id=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.client_id = client_id
        self.image_paths = []
        self.labels = []

        # ============================================================
        # 1. 确定当前模式对应的数据目录
        # ============================================================
        if mode == 'train':
            data_dir = os.path.join(root_dir, 'D_ID_train')
        elif mode == 'val':
            data_dir = os.path.join(root_dir, 'D_ID_val')
        elif mode == 'test':
            data_dir = os.path.join(root_dir, 'D_ID_test')
        elif mode == 'near_ood':
            data_dir = os.path.join(root_dir, 'D_Near_test')
        elif mode == 'far_ood':
            data_dir = os.path.join(root_dir, 'D_Far_test')
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ============================================================
        # 2. 建立标签映射 (保留原有逻辑)
        # ============================================================
        base_train_dir = os.path.join(root_dir, 'D_ID_train')
        if os.path.exists(base_train_dir):
            id_dirs = sorted([
                d for d in os.listdir(base_train_dir)
                if os.path.isdir(os.path.join(base_train_dir, d))
            ])
            self.class_to_idx = {dirname: idx for idx, dirname in enumerate(id_dirs)}
        else:
            print(f"Warning: Base train dir {base_train_dir} not found for label mapping")

        # ============================================================
        # 3. 加载图像数据 (新增缓存逻辑)
        # ============================================================
        # 定义缓存文件路径，例如: ./Plankton_OOD_Dataset/cache_train.pkl
        cache_file = os.path.join(root_dir, f"cache_{mode}.pkl")

        if os.path.exists(cache_file):
            # 命中缓存：直接加载，秒开
            print(f"Loading cached file list from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            self.image_paths = cache_data['paths']
            self.labels = cache_data['labels']
        else:
            # 未命中缓存：执行原来的扫描逻辑
            print(f"Scanning files for {mode} dataset (this may take a while)...")
            if os.path.exists(data_dir):
                for dir_name in os.listdir(data_dir):
                    class_dir = os.path.join(data_dir, dir_name)
                    if not os.path.isdir(class_dir):
                        continue

                    # 确定标签
                    current_label = -1
                    if mode in ['train', 'val', 'test']:
                        if dir_name in self.class_to_idx:
                            current_label = self.class_to_idx[dir_name]
                        else:
                            continue
                    else:
                        current_label = -1

                    # 加载图片
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                            img_path = os.path.join(class_dir, img_name)
                            self.image_paths.append(img_path)
                            self.labels.append(current_label)

                # 扫描完成后，保存到缓存
                print(f"Saving scanned list to {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump({'paths': self.image_paths, 'labels': self.labels}, f)
            else:
                print(f"Warning: Data directory {data_dir} does not exist")

        print(f"Loaded {len(self.image_paths)} images for {mode} dataset")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    """
    获取图像变换

    Args:
        image_size: 目标图像尺寸

    Returns:
        train_transform: 训练集变换
        val_transform: 验证集变换
    """
    # 训练集变换 - 包含数据增强
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集变换 - 不包含数据增强
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def partition_data(dataset, n_clients=10, alpha=0.1):
    """
    使用狄利克雷分布划分数据到多个客户端

    Args:
        dataset: 数据集对象
        n_clients: 客户端数量
        alpha: 狄利克雷分布参数，控制数据异质性

    Returns:
        client_indices: 每个客户端的数据索引列表
    """
    n_classes = len(ID_CLASSES)

    # 按类别组织数据索引
    class_indices = {i: [] for i in range(n_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label != -1:  # 只处理ID数据
            class_indices[label].append(idx)

    # 使用狄利克雷分布生成客户端数据分布
    client_indices = [[] for _ in range(n_clients)]

    for class_idx in range(n_classes):
        # 为每个类别生成客户端分布
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))

        # 获取当前类别的所有索引
        class_data = class_indices[class_idx]
        np.random.shuffle(class_data)

        # 根据比例分配数据到客户端
        proportions = (proportions * len(class_data)).astype(int)
        proportions[-1] = len(class_data) - np.sum(proportions[:-1])

        start_idx = 0
        for client_id in range(n_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(class_data[start_idx:end_idx])
            start_idx = end_idx

    # 打乱每个客户端的数据
    for client_id in range(n_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def create_federated_loaders(data_root, n_clients=10, alpha=0.1, batch_size=32, image_size=224):
    """
    创建联邦学习数据加载器

    Args:
        data_root: 数据根目录
        n_clients: 客户端数量
        alpha: 狄利克雷分布参数
        batch_size: 批次大小
        image_size: 图像尺寸

    Returns:
        client_loaders: 客户端数据加载器列表
        test_loader: 测试数据加载器
        near_ood_loader: Near-OOD数据加载器
        far_ood_loader: Far-OOD数据加载器
        inc_loader: IN-C (域泛化) 数据加载器
    """
    train_transform, val_transform = get_transforms(image_size)

    # 创建训练数据集
    train_dataset = PlanktonDataset(data_root, transform=train_transform, mode='train')

    # 划分数据到客户端
    client_indices = partition_data(train_dataset, n_clients, alpha)

    # 创建客户端数据加载器
    client_loaders = []
    for client_id in range(n_clients):
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices[client_id])
        client_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,     # 多进程加载数据
            pin_memory=True,   # 开启锁页内存，加速 CPU->GPU 传输
            persistent_workers=True # 避免每个epoch结束后销毁进程重建的开销
        )
        client_loaders.append(client_loader)
        print(f"客户端 {client_id}: {len(client_dataset)} 样本")

    # 创建测试和OOD数据加载器
    test_dataset = PlanktonDataset(data_root, transform=val_transform, mode='test')
    near_ood_dataset = PlanktonDataset(data_root, transform=val_transform, mode='near_ood')
    far_ood_dataset = PlanktonDataset(data_root, transform=val_transform, mode='far_ood')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    near_ood_loader = DataLoader(
        near_ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    far_ood_loader = DataLoader(
        far_ood_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # [新增] 创建 IN-C 加载器
    inc_loader = get_inc_loader(data_root, batch_size, image_size, severity=3)

    print(f"联邦学习数据加载器创建完成:")
    print(f"  - 客户端数量: {len(client_loaders)}")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - Near-OOD: {len(near_ood_dataset)} 样本")
    print(f"  - Far-OOD: {len(far_ood_dataset)} 样本")
    print(f"  - IN-C (OOD泛化测试): {len(inc_loader.dataset)} 样本")

    # [修改] 返回值增加 inc_loader (建议放在最后)
    return client_loaders, test_loader, near_ood_loader, far_ood_loader, inc_loader

# [新增类] 模拟海洋环境的图像腐蚀
class Corruptions:
    """
    模拟海洋环境的常见图像腐蚀 (IN-C 风格)
    用于评估 OOD 泛化能力 (Challenge 2)
    """
    @staticmethod
    def gaussian_blur(img, severity=1):
        # 模拟水体浑浊
        # severity 1-5 控制模糊程度
        kernel_sizes = [3, 5, 7, 9, 11]
        k = kernel_sizes[min(severity-1, 4)]
        return transforms.GaussianBlur(kernel_size=k, sigma=1.0)(img)

    @staticmethod
    def brightness(img, severity=1):
        # 模拟深海光照不足
        factors = [0.9, 0.8, 0.7, 0.6, 0.5]
        f = factors[min(severity-1, 4)]
        return transforms.functional.adjust_brightness(img, f)

    @staticmethod
    def gaussian_noise(img, severity=1):
        # 模拟传感器噪点
        variances = [0.01, 0.03, 0.05, 0.08, 0.1]
        v = variances[min(severity-1, 4)]

        # 需要先转 Tensor 再加噪
        if not isinstance(img, torch.Tensor):
            img_t = transforms.ToTensor()(img)
        else:
            img_t = img

        noise = torch.randn_like(img_t) * v
        img_noisy = torch.clamp(img_t + noise, 0, 1)

        # 转回 PIL Image 以适配后续 transform 流程 (视具体 pipeline 而定)
        return transforms.ToPILImage()(img_noisy)

# [新增函数] 获取 IN-C 加载器
def get_inc_loader(data_root, batch_size=32, image_size=224, severity=3):
    """
    创建 IN-C (域泛化) 测试加载器
    使用 ID 测试集数据，但应用 '水体浑浊' (高斯模糊) 等腐蚀变换
    """
    # 定义腐蚀变换链
    # 注意：Normalize 必须在最后
    inc_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # 在这里插入腐蚀操作，例如模拟 3 级浑浊
        transforms.Lambda(lambda x: Corruptions.gaussian_blur(x, severity=severity)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 复用 ID 测试集的数据路径，但使用 inc_transform
    inc_dataset = PlanktonDataset(data_root, transform=inc_transform, mode='test')

    inc_loader = DataLoader(
        inc_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    return inc_loader

if __name__ == "__main__":
    # 测试数据加载器
    data_root = "./data"

    try:
        client_loaders, test_loader, near_ood_loader, far_ood_loader, inc_loader = create_federated_loaders(
            data_root, n_clients=3, batch_size=4, image_size=224
        )

        # 测试一个批次
        for client_id, loader in enumerate(client_loaders):
            for images, labels in loader:
                print(f"客户端 {client_id} - 批次图像尺寸: {images.shape}")
                print(f"客户端 {client_id} - 批次标签: {labels}")
                break

    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("请确保数据集已正确划分并放置在指定目录")