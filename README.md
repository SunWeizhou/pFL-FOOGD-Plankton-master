# pFL-FOOGD: 基于联邦学习的海洋浮游生物图像识别

本项目实现了基于pFL-FOOGD（Personalized Federated Learning with Feature-Oriented Out-of-Distribution Generalization and Detection）框架的海洋浮游生物图像识别系统。该系统结合了FedRoD（处理非IID数据）和FOOGD（处理分布外泛化和检测）技术。

## 项目概述

### 核心特性

- **联邦学习框架**: 使用FedRoD策略处理客户端间的数据异质性
- **OOD检测与泛化**: 集成FOOGD模块，包含SAG（Stein Augmented Generalization）和SM3D（Score Matching）
- **高性能骨干网络**: 基于DenseNet-169的预训练模型
- **严格数据划分**: 根据Han等人的论文规范划分ID、Near-OOD和Far-OOD数据

### 技术架构

- **Backbone**: DenseNet-169（或DenseNet-121）
- **FL Strategy**: FedRoD（Federated Robust Decoupling）
- **OOD Modules**:
  - SAG: 使用KSD Loss进行领域泛化
  - SM3D: 训练评分模型进行OOD检测

## 项目结构

```
code/
├── 新项目工作文档.MD          # 项目详细规格说明
├── README.md                 # 项目说明（本文档）
├── requirements.txt          # Python依赖
├── data_utils.py             # 数据加载和联邦学习数据划分
├── models.py                 # FedRoD模型和FOOGD模块定义
├── client.py                 # 联邦学习客户端
├── server.py                 # 联邦学习服务端
├── train_federated.py        # 联邦学习训练主脚本
├── eval_utils.py             # 评估工具
├── test_pipeline.py          # 系统测试脚本
└── split_dataset.py          # 数据集划分脚本（保留用于数据准备）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

首先需要将原始的DYB-PlanktonNet数据集按照新文档的严格类别定义进行划分：

```bash
python split_dataset.py
```

**注意**: 数据集需要按照以下目录结构组织：

```
./data/
    ├── ID_images/           # 54个类别 (Train/Val/Test)
    ├── OOD_Near/            # 26个类别 (仅测试)
    └── OOD_Far/             # 12个类别 (仅测试)
```

### 3. 测试系统组件

运行测试脚本验证所有组件正常工作：

```bash
python test_pipeline.py
```

### 4. 开始联邦学习训练

```bash
python train_federated.py \
    --data_root ./Plankton_OOD_Dataset \
    --n_clients 5 \
    --alpha 5.0 \
    --communication_rounds 50 \
    --local_epochs 1 \
    --batch_size 64 \
    --model_type densenet121 \
    --use_foogd \
    --output_dir ./experiments_5.0
```
```
    # 为了安全，请把真实的 ANTHROPIC API Key 写入项目根目录下的 `.claude_env` 文件（不要提交到 git）。
    # 示例（放入 `.claude_env`）：
    # export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
    # export ANTHROPIC_AUTH_TOKEN="YOUR_KEY_HERE"   # 替换为你的真实 Key
    # export ANTHROPIC_MODEL="deepseek-chat"
    # export ANTHROPIC_SMALL_FAST_MODEL="deepseek-chat"

    # 本仓库已配置：当你打开终端或激活 conda `base` 时会自动从项目根目录的 `.claude_env` 读取这些变量（如果存在）。

```


### 5. 评估训练结果

训练完成后，实验结果将保存在 `./experiments` 目录下，包含：

- 训练曲线图
- 模型检查点
- 评估报告
- 配置文件和训练历史

## 详细说明

### 数据集划分

根据项目工作文档的严格类别定义：

- **ID Classes (54个)**: 目标标签 (0-53)，用于训练(80%)、验证(10%)、测试(10%)
- **Near-OOD Classes (26个)**: 仅用于测试（OOD评估）
- **Far-OOD Classes (12个)**: 仅用于测试（OOD评估）

### FedRoD模型架构

FedRoD包含两个分类头：

- **通用头 (Head_G)**: 由服务器聚合，学习通用知识
- **个性化头 (Head_P)**: 保持在客户端本地，学习客户端特定偏差

### FOOGD模块

FOOGD集成两个OOD组件：

- **SAG (Stein Augmented Generalization)**: 使用KSD Loss进行特征空间正则化
- **SM3D (Score Matching)**: 训练轻量级评分模型进行OOD检测

### 联邦学习设置

- **客户端数量**: 10个
- **数据划分**: 使用狄利克雷分布（α=0.1表示高异质性）
- **通信轮次**: 50-100轮
- **本地训练轮数**: 1-5轮

## 环境要求

### 硬件要求

- **GPU**: NVIDIA RTX 3060 (6GB VRAM) 或更高
- **内存**: 16GB RAM 或更高

### 软件要求

- Python 3.8+
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- seaborn

### 内存优化

- 使用批次大小16或32
- 支持混合精度训练
- 如果DenseNet-169导致内存不足，可切换到DenseNet-121

## 实验结果

训练脚本会生成以下输出：

- **训练曲线**: 损失和准确率随通信轮次的变化
- **OOD检测性能**: ID vs Near-OOD 和 ID vs Far-OOD 的AUROC和FPR95
- **模型检查点**: 每10轮保存一次，包含最佳模型
- **评估报告**: 详细的分类和OOD检测指标

## 参数调优

### 训练参数

- `--alpha`: 狄利克雷分布参数，控制数据异质性（默认0.1）
- `--local_epochs`: 客户端本地训练轮数（默认1）
- `--client_fraction`: 每轮选择的客户端比例（默认1.0）

### 模型参数

- `--model_type`: 骨干网络类型（densenet121或densenet169）
- `--use_foogd`: 是否启用FOOGD模块
- `--batch_size`: 批次大小（默认32）

## 故障排除

1. **数据集不存在**: 确保已运行 `split_dataset.py` 并正确设置数据路径
2. **内存不足**: 减小批次大小或切换到DenseNet-121
3. **模型加载失败**: 检查PyTorch版本兼容性
4. **训练不稳定**: 调整学习率或增加本地训练轮数

## 引用

本项目基于以下技术实现：

- FedRoD: "Federated Robust Decoupling"
- FOOGD: "Feature-Oriented Out-of-Distribution Generalization and Detection"
- Han et al. - "Benchmarking Out-of-Distribution Detection for Plankton Recognition"

