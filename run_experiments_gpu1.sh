#!/bin/bash

# ========================================================
# pFL-FOOGD 全面基准测试 - GPU 1 (FedRoD系列/核心组)
# 包含 6 组实验 (FOOGD默认开启，对比 Taxonomy 效果)：
# 1. FedRoD + FOOGD (Strong Baseline)
# 2. TR-FedRoD + FOOGD (Ours)
# 覆盖 Alpha 0.1, 0.5, 5.0
# ========================================================

export CUDA_VISIBLE_DEVICES=1

# 基础配置
DATA_ROOT="./Plankton_OOD_Dataset"
N_CLIENTS=10
ROUNDS=100
EPOCHS=3  # 这里的 EPOCHS 对应 local_epochs
BATCH_SIZE=64
IMAGE_SIZE=299
MODEL="densenet121"
SEED=2025

# 创建日志目录
mkdir -p logs_benchmark

echo "========================================================"
echo "开始运行 FedRoD 系列全面实验 - GPU 1 (共 6 组)"
echo "变量: Taxonomy (关 vs 开) | 固定: FOOGD=True"
echo "Alpha设置: 0.1 (重点), 0.5, 5.0"
echo "开始时间: $(date)"
echo "========================================================"

# 定义实验运行函数
run_experiment() {
    local ALPHA=$1
    local USE_FOOGD=$2
    local ALGORITHM=$3
    local EXP_NAME=$4
    local DESC=$5
    local USE_TAXONOMY=$6  # [新增] 第6个参数控制 Taxonomy

    echo ""
    echo "--------------------------------------------------------"
    echo "正在运行: $EXP_NAME"
    echo "场景: $DESC"
    echo "配置: Alpha=$ALPHA | Algo=$ALGORITHM | FOOGD=$USE_FOOGD | Tax=$USE_TAXONOMY"
    echo "--------------------------------------------------------"

    CMD="python train_federated.py \
        --data_root $DATA_ROOT \
        --n_clients $N_CLIENTS \
        --alpha $ALPHA \
        --communication_rounds $ROUNDS \
        --local_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --image_size $IMAGE_SIZE \
        --model_type $MODEL \
        --seed $SEED \
        --algorithm $ALGORITHM \
        --output_dir ./experiments_benchmark/$EXP_NAME"

    if [ "$USE_FOOGD" = "true" ]; then
        CMD="$CMD --use_foogd"
    fi

    # [新增] Taxonomy 开关逻辑
    if [ "$USE_TAXONOMY" = "true" ]; then
        CMD="$CMD --use_taxonomy"
    fi

    # 运行并记录日志
    $CMD 2>&1 | tee "logs_benchmark/${EXP_NAME}.log"
}

# =================== 实验队列 ===================

# --- 第一轮：Alpha = 0.1 (极端异质性 - 论文核心对比) ---
# Strong Baseline: FedRoD + FOOGD
run_experiment 0.1 "true" "fedrod" "fedrod_foogd_alpha0.1_base" "FedRoD+FOOGD (Base, Alpha=0.1)" "false"
# Ours: TR-FedRoD + FOOGD
run_experiment 0.1 "true" "fedrod" "fedrod_foogd_alpha0.1_tax"  "TR-FedRoD+FOOGD (Ours, Alpha=0.1)" "true"

# --- 第二轮：Alpha = 0.5 (真实强异质性) ---
run_experiment 0.5 "true" "fedrod" "fedrod_foogd_alpha0.5_base" "FedRoD+FOOGD (Base, Alpha=0.5)" "false"
run_experiment 0.5 "true" "fedrod" "fedrod_foogd_alpha0.5_tax"  "TR-FedRoD+FOOGD (Ours, Alpha=0.5)" "true"

# --- 第三轮：Alpha = 5.0 (均匀分布) ---
run_experiment 5.0 "true" "fedrod" "fedrod_foogd_alpha5.0_base" "FedRoD+FOOGD (Base, Alpha=5.0)" "false"
run_experiment 5.0 "true" "fedrod" "fedrod_foogd_alpha5.0_tax"  "TR-FedRoD+FOOGD (Ours, Alpha=5.0)" "true"

echo "========================================================"
echo "GPU 1 所有 6 组实验已完成！"
echo "结束时间: $(date)"
echo "========================================================"