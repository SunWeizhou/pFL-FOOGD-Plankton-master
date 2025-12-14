#!/bin/bash

# ========================================================
# pFL-FOOGD 全面基准测试 - GPU 0 (FedAvg系列)
# 包含 6 组实验：
# 1. FedAvg (Alpha 0.1, 0.5, 5.0)
# 2. FedAvg + FOOGD (Alpha 0.1, 0.5, 5.0)
# ========================================================

export CUDA_VISIBLE_DEVICES=0

# 基础配置
DATA_ROOT="./Plankton_OOD_Dataset"
N_CLIENTS=10
ROUNDS=50
EPOCHS=3
BATCH_SIZE=64
IMAGE_SIZE=256
MODEL="densenet121"
SEED=2025

# 创建日志目录
mkdir -p logs_benchmark

echo "========================================================"
echo "开始运行 FedAvg 系列全面实验 - GPU 0 (共 6 组)"
echo "Alpha设置: 0.1 (极端), 0.5 (真实), 5.0 (均匀)"
echo "开始时间: $(date)"
echo "========================================================"

# 定义实验运行函数
run_experiment() {
    local ALPHA=$1
    local USE_FOOGD=$2
    local ALGORITHM=$3
    local EXP_NAME=$4
    local DESC=$5

    echo ""
    echo "--------------------------------------------------------"
    echo "正在运行: $EXP_NAME"
    echo "场景: $DESC"
    echo "配置: Alpha=$ALPHA | Algorithm=$ALGORITHM | FOOGD=$USE_FOOGD"
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

    # 运行并记录日志
    $CMD 2>&1 | tee "logs_benchmark/${EXP_NAME}.log"
}

# =================== 实验队列 ===================

# --- 第一轮：Alpha = 0.1 (极端异质性) ---
run_experiment 0.1 "false" "fedavg" "fedavg_alpha0.1" "FedAvg (Alpha=0.1)"
run_experiment 0.1 "true"  "fedavg" "fedavg_foogd_alpha0.1" "FedAvg+FOOGD (Alpha=0.1)"

# --- 第二轮：Alpha = 0.5 (真实强异质性 - Sweet Spot) ---
run_experiment 0.5 "false" "fedavg" "fedavg_alpha0.5" "FedAvg (Alpha=0.5)"
run_experiment 0.5 "true"  "fedavg" "fedavg_foogd_alpha0.5" "FedAvg+FOOGD (Alpha=0.5)"

# --- 第三轮：Alpha = 5.0 (中等/均匀分布) ---
run_experiment 5.0 "false" "fedavg" "fedavg_alpha5.0" "FedAvg (Alpha=5.0)"
run_experiment 5.0 "true"  "fedavg" "fedavg_foogd_alpha5.0" "FedAvg+FOOGD (Alpha=5.0)"

echo "========================================================"
echo "GPU 0 所有 6 组实验已完成！"
echo "结束时间: $(date)"
echo "========================================================"