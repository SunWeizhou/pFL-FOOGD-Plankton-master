# pFL-FOOGD 系统架构与代码详解

## 📋 项目概述

pFL-FOOGD (Personalized Federated Learning with Feature-Oriented Out-of-Distribution Generalization and Detection) 是一个结合了FedRoD（联邦鲁棒解耦）和FOOGD（面向特征的OOD泛化与检测）技术的联邦学习框架，专门用于海洋浮游生物图像识别任务。

## 🏗️ 系统架构总览

### 整体架构图
```
┌─────────────────────────────────────────────────────────────┐
│                   联邦学习服务器 (FLServer)                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 全局模型: DenseNet-121 + Head_G (通用头)              │  │
│  │ 功能: 模型聚合、客户端管理、评估协调                    │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ 广播模型 / 收集更新
┌───────────────────────────┼─────────────────────────────────┐
│                   联邦学习客户端 (FLClient)                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 本地模型: 全局模型副本 + Head_P (个性化头)            │  │
│  │ FOOGD模块: ScoreModel + KSD损失 + SM3D损失           │  │
│  │ 功能: 本地训练、OOD检测、个性化学习                    │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计理念
1. **模型解耦**: Head_G学习全局知识，Head_P学习客户端特定知识
2. **OOD处理**: FOOGD模块同时提升OOD泛化能力和检测能力
3. **非IID适应**: Dirichlet分布模拟真实数据异质性
4. **个性化联邦**: 客户端获得个性化模型，同时参与全局学习

## 🔧 代码架构详解

### 1. 模型定义模块 (`models.py`, 363行)

#### 1.1 DenseNetBackbone类
```python
class DenseNetBackbone(nn.Module):
    """DenseNet骨干网络 - 特征提取器"""
```
- **功能**: 基于DenseNet-121/169的特征提取器
- **关键设计**:
  - 移除最后的分类层，输出固定维度特征向量
  - 支持预训练权重，加速收敛
  - 特征维度: DenseNet-121(1024), DenseNet-169(1664)
- **前向传播**: 输入图像 → 特征向量

#### 1.2 FedRoD_Model类
```python
class FedRoD_Model(nn.Module):
    """FedRoD模型 - 包含通用头和个性化头"""
```
- **双头架构设计**:
  - `Head_G` (通用头): 服务器聚合，学习全局知识
  - `Head_P` (个性化头): 客户端本地保持，学习客户端特定偏差
- **参数管理**:
  - Head_G参数参与联邦聚合
  - Head_P参数始终保持在客户端本地
- **前向传播**: 输入图像 → (logits_g, logits_p, features)

#### 1.3 FOOGD_Module类
```python
class FOOGD_Module(nn.Module):
    """FOOGD模块 - 集成SAG和SM3D"""
```
- **核心组件**:
  - `ScoreModel`: 轻量级评分网络，学习特征空间的梯度
  - `compute_sm3d_loss()`: Score Matching损失（DSM + MMD）
  - `compute_ksd_loss()`: 核化Stein差异损失（已优化：归一化除以特征维度）
  - `rbf_kernel_matrix()`: RBF核矩阵计算，支持预计算优化

- **OOD检测原理**:
  - OOD样本的评分范数（梯度大小）更大，表示密度更低
  - 训练评分模型区分ID和OOD特征分布

- **关键优化**:
  - KSD损失归一化：`ksd / dim`，与分类损失尺度匹配
  - 内存优化：单次距离矩阵计算
  - 数值稳定性：防止梯度爆炸

#### 1.4 工厂函数
```python
def create_fedrod_model(model_type='densenet169', num_classes=54, use_foogd=True):
```
- **功能**: 创建完整的FedRoD模型和FOOGD模块
- **参数配置**: 支持不同骨干网络、类别数量、FOOGD开关

### 2. 数据加载模块 (`data_utils.py`, 430行)

#### 2.1 严格类别定义
```python
ID_CLASSES = [...]        # 54个ID类（训练/测试）
NEAR_OOD_CLASSES = [...]  # 26个Near-OOD类（仅测试）
FAR_OOD_CLASSES = [...]   # 12个Far-OOD类（仅测试）
```
- **数据分离**: 确保训练集和测试集无数据泄漏
- **OOD分类**: 明确定义Near-OOD和Far-OOD类别

#### 2.2 PlanktonDataset类
```python
class PlanktonDataset(Dataset):
    """浮游生物数据集类 - 支持缓存加速"""
```
- **四种数据模式**:
  - `train`: D_ID_train (ID训练集)
  - `test`: D_ID_test (ID测试集)
  - `near_ood`: D_Near_test (Near-OOD测试集)
  - `far_ood`: D_Far_test (Far-OOD测试集)

- **数据增强**:
  - 训练时: 随机裁剪、翻转、颜色抖动
  - 测试时: 中心裁剪、标准化

#### 2.3 联邦数据分区
```python
def create_federated_datasets(...):
    """使用Dirichlet分布创建非IID的客户端数据集"""
```
- **Dirichlet分布**: 参数α控制数据异质性程度
  - α=0.1: 极端异质性（完全隔离站点）
  - α=0.5: 真实强异质性（典型盐度梯度差异）
  - α=5.0: 中等异质性
  - α=10.0: 接近IID（理想均匀混合）

- **性能优化**:
  - 预计算类-索引映射，加速客户端数据采样
  - 支持缓存，减少磁盘I/O

### 3. 联邦学习客户端 (`client.py`, 342行)

#### 3.1 FLClient类
```python
class FLClient:
    """联邦学习客户端实现"""
```

#### 3.2 初始化过程
```python
def __init__(self, client_id, train_dataset, test_dataset, ...):
```
- **本地数据**: 分配非IID训练集和测试集
- **模型副本**: 创建全局模型的本地副本
- **优化器配置**:
  - 主模型: SGD优化器（骨干网络 + Head_G + Head_P）
  - FOOGD模块: Adam优化器（ScoreModel）

#### 3.3 训练流程
```python
def local_train(self, global_model, foogd_module=None):
```
1. **模型更新**: 加载全局模型参数
2. **本地训练循环**:
   - 前向传播: 计算分类损失
   - FOOGD训练（如果启用）:
     - 计算KSD损失（特征空间正则化）
     - 计算SM3D损失（评分模型训练）
   - 总损失: `分类损失 + λ_ksd*KSD损失 + λ_sm*SM3D损失`
   - 反向传播和参数更新

3. **关键参数**:
   - `target_lambda_ksd = 0.01` (KSD损失权重)
   - `target_lambda_sm = 0.1` (SM3D损失权重)

#### 3.4 模型更新策略
- **上传参数**: 只上传Head_G参数到服务器
- **本地保持**: Head_P参数始终保持在客户端
- **FOOGD参数**: ScoreModel参数本地训练，不参与联邦聚合

### 4. 联邦学习服务器 (`server.py`, 266行)

#### 4.1 FLServer类
```python
class FLServer:
    """联邦学习服务器实现"""
```

#### 4.2 初始化过程
```python
def __init__(self, data_root, n_clients=5, ...):
```
- **全局模型**: 创建FedRoD模型
- **客户端初始化**: 创建FLClient列表，分配数据
- **评估设置**: 准备ID和OOD测试集

#### 4.3 联邦训练循环
```python
def federated_train(self, communication_rounds=50, ...):
```
1. **客户端选择**: 每轮选择部分客户端参与训练（`client_fraction`参数）
2. **模型广播**: 发送全局模型（骨干网络 + Head_G）到选定客户端
3. **本地训练**: 客户端进行本地训练
4. **更新收集**: 收集客户端的Head_G更新
5. **模型聚合**: 加权平均聚合更新，权重基于客户端样本数量

#### 4.4 评估与监控
- **定期评估**: 每`eval_frequency`轮评估一次
- **性能指标**: ID准确率、OOD检测AUROC/FPR95
- **检查点管理**: 定期保存模型，支持训练恢复

### 5. 主训练脚本 (`train_federated.py`, 624行)

#### 5.1 参数解析系统
```python
def parse_args():
```
- **丰富参数**: 支持30+个可配置参数
- **实验管理**: 输出目录、随机种子、检查点频率
- **性能优化**: 批次大小、图像尺寸、工作进程数

#### 5.2 训练流程控制
```python
def main():
```
1. **环境设置**: 设备检测、随机种子固定
2. **数据准备**: 加载数据集，创建联邦分区
3. **模型初始化**: 创建FedRoD模型和FOOGD模块
4. **服务器初始化**: 创建FLServer，管理客户端
5. **联邦训练循环**: 执行指定轮次的联邦学习
6. **评估与保存**: 最终评估，保存模型和结果

#### 5.3 性能优化特性
- **预生成测试加载器**: 避免重复创建DataLoader
- **持久化工作进程**: `persistent_workers=True`提高数据加载效率
- **混合精度训练**: 支持`torch.amp`减少内存使用
- **梯度裁剪**: 防止梯度爆炸

### 6. 评估工具 (`eval_utils.py`)

#### 6.1 综合评估函数
```python
def evaluate_model(...):
```
- **ID分类评估**: 准确率、混淆矩阵、类别准确率
- **OOD检测评估**: AUROC、FPR95、ROC曲线
- **个性化增益**: Head_P vs Head_G准确率差异

#### 6.2 可视化功能
- **训练曲线**: 损失和准确率变化趋势
- **混淆矩阵**: ID分类性能可视化
- **ROC曲线**: OOD检测性能可视化
- **综合对比**: ID-OOD性能平衡关系

### 7. 实验管理脚本

#### 7.1 并行实验系统
- `run_experiments_gpu0.sh`: GPU 0实验脚本（α=0.1, 0.5）
- `run_experiments_gpu1.sh`: GPU 1实验脚本（α=5.0）
- `run_experiments_parallel.sh`: 并行启动脚本

#### 7.2 实验配置管理
- **参数一致性**: 固定随机种子确保可重复性
- **日志系统**: 完整的实验日志记录
- **错误处理**: 进程监控和错误恢复

## 🎯 模型设计创新点

### 1. FedRoD双头架构创新
- **知识解耦**: Head_G学习全局模式，Head_P学习本地偏差
- **参数隔离**: 只有Head_G参与联邦聚合，保护客户端隐私
- **个性化优势**: 客户端获得定制化模型，提升本地性能

### 2. FOOGD模块设计创新
- **双重OOD处理**: 同时提升OOD泛化（SAG）和检测（SM3D）
- **特征空间正则化**: KSD损失对齐特征分布，提高泛化能力
- **轻量级评分模型**: 高效学习特征空间梯度，实现OOD检测

### 3. 非IID数据模拟创新
- **Dirichlet分布**: 参数化控制数据异质性程度
- **现实场景模拟**: α=0.5模拟真实盐度梯度差异
- **全面实验设计**: 覆盖极端异质性到接近IID的完整谱系

### 4. 性能优化创新
- **KSD损失归一化**: 解决损失尺度不匹配问题
- **内存优化计算**: 单次距离矩阵计算减少内存占用
- **预计算加速**: 类-索引映射加速数据采样

## 📊 实验配置总结

### 当前实验设置
```python
# 基础配置
model_type = "densenet121"      # 骨干网络
n_clients = 5                   # 客户端数量
communication_rounds = 100      # 通信轮数
local_epochs = 3                # 本地训练轮数
batch_size = 64                 # 批次大小
image_size = 299                # 图像尺寸
seed = 2025                     # 随机种子

# FOOGD参数
target_lambda_ksd = 0.01        # KSD损失权重
target_lambda_sm = 0.1          # SM3D损失权重

# 数据异质性参数
alpha_values = [0.1, 0.5, 5.0]  # Dirichlet分布参数
```

### 已完成实验
| 实验名称 | α值 | FOOGD | 描述 | 状态 |
|----------|-----|-------|------|------|
| alpha0.1_with_foogd | 0.1 | ✅ | 极端异质性 (With FOOGD) | ✅ 完成 |
| alpha0.1_no_foogd | 0.1 | ❌ | 极端异质性 (Baseline) | ✅ 完成 |
| alpha0.5_with_foogd | 0.5 | ✅ | 真实强异质性 (With FOOGD) | ✅ 完成 |
| alpha0.5_no_foogd | 0.5 | ❌ | 真实强异质性 (Baseline) | ✅ 完成 |
| alpha5.0_with_foogd | 5.0 | ✅ | 中等异质性 (With FOOGD) | ✅ 完成 |
| alpha5.0_no_foogd | 5.0 | ❌ | 中等异质性 (Baseline) | ✅ 完成 |

## 🔧 代码使用指南

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 数据集准备
```bash
python split_dataset.py
```

### 3. 单次实验运行
```bash
python train_federated.py \
    --data_root ./Plankton_OOD_Dataset \
    --n_clients 5 \
    --alpha 0.5 \
    --communication_rounds 100 \
    --local_epochs 3 \
    --batch_size 64 \
    --image_size 299 \
    --model_type densenet121 \
    --seed 2025 \
    --use_foogd \
    --output_dir ./experiments/test
```

### 4. 批量实验运行
```bash
# 并行运行所有实验
bash run_experiments_parallel.sh

# 单独运行GPU实验
./run_experiments_gpu0.sh
./run_experiments_gpu1.sh
```

### 5. 结果可视化
```bash
# 重新生成可视化图像
python visualize_experiments.py \
    --experiment_dirs experiments/alpha0.1_with_foogd \
                     experiments/alpha0.1_no_foogd \
                     experiments/alpha0.5_with_foogd \
                     experiments/alpha0.5_no_foogd \
                     experiments/alpha5.0_with_foogd \
                     experiments/alpha5.0_no_foogd
```

## 📈 性能表现总结

基于6组完整实验的结果：

### 1. FOOGD效果验证
- **OOD检测AUROC提升**: 43-57%（从~40%提升到87-94%）
- **OOD检测FPR95降低**: 超过50个百分点（从80-95%降至24-38%）
- **ID分类准确率**: 小幅提升0.10-0.58%（证明不损害ID性能）

### 2. 推荐部署配置
**`alpha0.5_with_foogd`**:
- **现实性**: α=0.5模拟真实盐度梯度差异
- **性能**: ID准确率96.12%，OOD检测AUROC 92.31%
- **个性化增益**: 2.0-2.5%，客户端受益明显
- **稳定性**: 训练收敛稳定，误报率低（FPR95: 24-27%）

## 🚀 扩展与优化方向

### 1. 模型架构扩展
- 测试DenseNet-169、ResNet等不同骨干网络
- 探索更复杂的双头架构设计
- 优化FOOGD模块的评分模型架构

### 2. 算法优化
- 自适应λ参数调整策略
- 改进的KSD损失计算方法
- 更高效的OOD检测算法

### 3. 实验扩展
- 更多α值测试（0.01, 0.2, 1.0, 2.0, 10.0）
- 不同客户端数量影响分析
- 长期训练稳定性测试

### 4. 实际部署
- 实时OOD检测接口开发
- 模型压缩和加速优化
- 多节点联邦学习部署

---

*文档更新: 2025年12月11日*
*代码版本: pFL-FOOGD v2.0*
*实验状态: 6组实验全部完成*