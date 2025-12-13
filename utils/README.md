# 工具脚本目录 (utils/)

这个目录包含了各种辅助脚本，用于验证、测试和评估pFL-FOOGD系统。

## 验证脚本 (Verification)

### `check_md5_leakage.py`
**功能**：通过MD5哈希检查训练集和测试集是否有数据泄漏
**用途**：确保数据划分的正确性
**使用方法**：
```bash
python check_md5_leakage.py
```

### `check_overlap.py`
**功能**：通过文件名检查训练集和测试集是否有重叠
**用途**：快速验证数据划分
**使用方法**：
```bash
python check_overlap.py
```

### `verify_leakage.py`
**功能**：综合验证数据泄漏（结合多种检查方法）
**用途**：最全面的数据泄漏检查
**使用方法**：
```bash
python verify_leakage.py
```

## 测试脚本 (Testing)

### `test_pipeline.py`
**功能**：单元测试脚本，验证各个组件的功能
**测试内容**：
- 模型创建和前向传播
- 数据加载器
- 客户端和服务器功能
- 评估工具
**使用方法**：
```bash
python test_pipeline.py
```

### `test_foogd_pipeline.py`
**功能**：集成测试脚本，运行小规模实验验证FOOGD模块
**特点**：
- 少量客户端（3个）
- 少量通信轮数（5轮）
- 小批量大小（16）
- 对比使用/不使用FOOGD的效果
**使用方法**：
```bash
python test_foogd_pipeline.py
```

## 评估脚本 (Evaluation)

### `evaluate_head_p.py`
**功能**：个性化模型评估脚本
**特点**：
- 加载已保存的`best_model.pth`
- 评估每个客户端的个性化Head-P性能
- 对比Head-G（全局头）和Head-P（个性化头）
- 计算个性化增益（Personalization Gain）
**使用方法**：
```bash
python evaluate_head_p.py --experiments_dir ../experiments --data_root ../Plankton_OOD_Dataset
```

### `evaluate_inc_robustness.py`
**功能**：IN-C（ImageNet-C风格）鲁棒性评估
**特点**：
- 测试三种腐蚀类型：高斯模糊、高斯噪声、亮度变化
- 5个严重程度等级（1-5）
- 生成Markdown格式的报告表格
- 计算平均准确率下降（mCE Drop）
**使用方法**：
```bash
python evaluate_inc_robustness.py --model_path ../experiments/alpha0.1_no_foogd/experiment_20251210_024137/best_model.pth --batch_size 32 --image_size 224
```

## 使用建议

1. **项目开发阶段**：使用验证脚本确保数据划分正确
2. **代码修改后**：使用测试脚本验证组件功能
3. **实验完成后**：使用评估脚本进行详细分析和生成报告
4. **论文写作时**：使用评估脚本生成可直接复制的表格和数据