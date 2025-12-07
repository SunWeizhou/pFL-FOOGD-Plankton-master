#!/bin/bash

# ========================================================
# pFL-FOOGD 自动化实验脚本 (v2.0)
# 对应场景：极端差异(0.1)、真实强差异(0.5)、真实中差异(1.0)、基准对照(10.0)
# ========================================================

# 1. 基础配置 (请根据服务器实际情况调整)
DATA_ROOT="./Plankton_OOD_Dataset"
N_CLIENTS=5
ROUNDS=50
EPOCHS=1
BATCH_SIZE=64
MODEL="densenet169"     # 如果显存充足可改为 densenet169
SEED=2025               # 固定随机种子，确保所有实验的数据划分完全一致！

# 创建日志目录
mkdir -p logs

echo "========================================================"
echo "开始运行 pFL-FOOGD 全面对照实验 (共 8 组)"
echo "开始时间: $(date)"
echo "随机种子: $SEED (保证数据划分一致性)"
echo "========================================================"

# 定义单次实验运行函数
run_experiment() {
    local ALPHA=$1
    local USE_FOOGD=$2
    local EXP_NAME=$3
    local DESC=$4

    echo ""
    echo "--------------------------------------------------------"
    echo "正在运行: $EXP_NAME"
    echo "场景说明: $DESC"
    echo "配置: Alpha=$ALPHA | Use FOOGD=$USE_FOOGD"
    echo "--------------------------------------------------------"

    # 构建基础命令
    # 注意：这里假设你已经修改了 train_federated.py，增加了 --seed 参数
# 在这里定义新的尺寸变量
    IMG_SIZE=320 

    CMD="python train_federated.py \
        --data_root $DATA_ROOT \
        --n_clients $N_CLIENTS \
        --alpha $ALPHA \
        --communication_rounds $ROUNDS \
        --local_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --image_size $IMG_SIZE \
        --model_type $MODEL \
        --seed $SEED \
        --output_dir ./experiments/$EXP_NAME"

    # 根据开关添加 --use_foogd 参数
    # 注意：这里假设你已经将 train_federated.py 中 use_foogd 的 default 改为了 False
    if [ "$USE_FOOGD" = "true" ]; then
        CMD="$CMD --use_foogd"
    fi

    # 执行命令并保存日志
    # 2>&1 | tee ... 会同时在屏幕显示和写入文件，方便你实时查看进度
    echo "执行命令: $CMD"
    $CMD > "logs/${EXP_NAME}.log" 2>&1

    echo ">>> 实验 $EXP_NAME 完成！"
}

# ================= 实验队列 (共8组) =================

# --- 第1组：极端差异 (Alpha=0.1) ---
# 意义：模拟完全隔离的站点（如远海 vs 淡水），验证算法在恶劣条件下的鲁棒性下界。
run_experiment 0.1 "true"  "alpha0.1_with_foogd" "极端异质性 (With FOOGD)"
run_experiment 0.1 "false" "alpha0.1_no_foogd"   "极端异质性 (Baseline)"

# --- 第2组：真实强差异 (Alpha=0.5) [核心组] ---
# 意义：模拟珠三角典型的盐度梯度差异，优势种不同但有少量重叠。这是最符合实际的场景。
run_experiment 0.5 "true"  "alpha0.5_with_foogd" "真实强异质性 (With FOOGD)"
run_experiment 0.5 "false" "alpha0.5_no_foogd"   "真实强异质性 (Baseline)"

# --- 第3组：真实中差异 (Alpha=1.0) ---
# 意义：模拟水流交换频繁的河口混合区，物种分布相对均匀但仍有偏差。
run_experiment 1.0 "true"  "alpha1.0_with_foogd" "真实中异质性 (With FOOGD)"
run_experiment 1.0 "false" "alpha1.0_no_foogd"   "真实中异质性 (Baseline)"

# --- 第4组：理想对照 (Alpha=10.0) ---
# 意义：模拟理想均匀混合（接近 IID），作为系统性能的上界基准。
run_experiment 10.0 "true"  "alpha10_with_foogd" "IID对照组 (With FOOGD)"
run_experiment 10.0 "false" "alpha10_no_foogd"   "IID对照组 (Baseline)"

echo "========================================================"
echo "所有 8 组实验已全部完成！请检查 logs/ 目录下的日志文件。"
echo "结束时间: $(date)"
echo "========================================================"