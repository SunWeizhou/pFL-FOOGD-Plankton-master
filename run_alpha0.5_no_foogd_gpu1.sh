#!/bin/bash

# ========================================================
# pFL-FOOGD 单实验脚本 - GPU 1
# 运行 alpha=0.5_no_foogd 实验
# ========================================================

# 设置使用显卡1
export CUDA_VISIBLE_DEVICES=1

# 基础配置
DATA_ROOT="./Plankton_OOD_Dataset"
N_CLIENTS=5
ROUNDS=100
EPOCHS=3
BATCH_SIZE=64
MODEL="densenet121"
SEED=2025
IMG_SIZE=299

# 创建日志目录
mkdir -p logs

echo "========================================================"
echo "开始运行 pFL-FOOGD 实验 - GPU 1"
echo "实验: alpha0.5_no_foogd (真实强异质性 Baseline)"
echo "开始时间: $(date)"
echo "随机种子: $SEED (保证数据划分一致性)"
echo "使用显卡: GPU 1 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "========================================================"

# 构建命令
CMD="python train_federated.py \
    --data_root $DATA_ROOT \
    --n_clients $N_CLIENTS \
    --alpha 0.5 \
    --communication_rounds $ROUNDS \
    --local_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --image_size $IMG_SIZE \
    --model_type $MODEL \
    --seed $SEED \
    --compute_aug_features \
    --freeze_bn \
    --output_dir ./experiments/alpha0.5_no_foogd"

# 注意：这里不添加 --use_foogd 参数，因为这是 baseline 实验

echo "执行命令: $CMD"
echo ""

# 执行命令并保存日志
$CMD 2>&1 | tee "logs/alpha0.5_no_foogd_gpu1.log"

echo ""
echo "========================================================"
echo "实验 alpha0.5_no_foogd 已完成！"
echo "结束时间: $(date)"
echo "日志文件: logs/alpha0.5_no_foogd_gpu1.log"
echo "输出目录: experiments/alpha0.5_no_foogd"
echo "========================================================"