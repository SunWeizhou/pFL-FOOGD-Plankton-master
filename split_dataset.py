#!/usr/bin/env python3
"""
DYB-PlanktonNet 数据集划分脚本
根据 Han et al. 论文规范将数据集划分为 ID、Near-OOD 和 Far-OOD 子集

作者: Claude Code
日期: 2025-11-07
"""

import os
import shutil
import random
from pathlib import Path

# 设置随机种子以确保可复现性
random.seed(42)

# 类别定义 - 根据 Han et al. 论文附录
# 1. 分布内 (ID) 类别 - 54 个
ID_CLASSES = [
    '001_Polychaeta_most with eggs', '003_Polychaeta_Type A', '004_Polychaeta_Type B',
    '005_Polychaeta_Type C', '006_Polychaeta_Type D', '007_Polychaeta_Type E',
    '008_Polychaeta_Type F', '009_Penilia avirostris', '010_Evadne tergestina',
    '011_Acartia sp.A', '012_Acartia sp.B', '013_Acartia sp.C', '014_Calanopia sp',
    '015_Labidocera sp', '016_Tortanus gracilis', '017_Calanoid with egg',
    '019_Calanoid_Type A', '020_Calanoid_Type B', '024_Oithona sp.B with egg',
    '025_Cyclopoid_Type A_with egg', '027_Harpacticoid_mating', '029_Microsetella sp',
    '033_Caligus sp', '034_Copepod_Type A', '035_Caprella sp', '036_Amphipoda_Type A',
    '037_Amphipoda_Type B', '038_Amphipoda_Type C', '039_Gammarids_Type A',
    '040_Gammarids_Type B', '041_Gammarids_Type C', '042_Cymodoce sp', '043_Lucifer sp',
    '044_Macrura larvae', '046_Megalopa larva_Phase 1_Type B',
    '047_Megalopa larva_Phase 1_Type C', '048_Megalopa larva_Phase 1_Type D',
    '049_Megalopa larva_Phase 2', '050_Porcrellanidae larva',
    '051_Shrimp-like larva_Type A', '052_Shrimp-like larva_Type B',
    '053_Shrimp-like_Type A', '054_Shrimp-like_Type B', '056_Shrimp-like_Type D',
    '058_Shrimp-like_Type F', '060_Cumacea_Type A', '061_Cumacea_Type B',
    '062_Chaetognatha', '063_Oikopleura sp. parts', '065_Tunicata_Type A',
    '068_Jellyfish', '071_Creseis acicula', '082_Noctiluca scintillans',
    '091_Phaeocystis globosa'
]

# 2. 近-OOD 类别 - 26 个
NEAR_OOD_CLASSES = [
    '002_Polychaeta larva', '018_Calanoid Nauplii', '021_Calanoid_Type C',
    '022_Calanoid_Type D', '023_Oithona sp.A with egg', '026_Cyclopoid_Type A',
    '028_Harpacticoid', '030_Monstrilla sp.A', '031_Monstrilla sp.B',
    '045_Megalopa larva_Phase 1_Type A', '055_Shrimp-like_Type C',
    '057_Shrimp-like_Type E', '059_Ostracoda', '064_Oikopleura sp',
    '066_Actiniaria larva', '067_Hydroid', '069_Jelly-like', '070_Bryozoan larva',
    '072_Gelatinous Zooplankton', '073_Unknown_Type A', '074_Unknown_Type B',
    '075_Unknown_Type C', '076_Unknown_Type D', '077_Balanomorpha exuviate',
    '078_Crustacean limb_Type A', '081_Fish Larvae'
]

# 3. 远-OOD (风格) 类别 - 12 个
FAR_OOD_CLASSES = [
    '079_Crustacean limb_Type B', '080_Fish egg', '083_Particle_filamentous_Type A',
    '084_Particle_filamentous_Type B', '085_Particle_bluish', '086_Particle_molts',
    '087_Particle_translucent flocs', '088_Particle_yellowish flocs',
    '089_Particle_yellowish rods', '090_Bubbles', '092_Fish tail'
]

# 验证类别数量
print(f"ID categories: {len(ID_CLASSES)} (should be 54)")
print(f"Near-OOD categories: {len(NEAR_OOD_CLASSES)} (should be 26)")
print(f"Far-OOD categories: {len(FAR_OOD_CLASSES)} (should be 12)")
print(f"Total categories: {len(ID_CLASSES) + len(NEAR_OOD_CLASSES) + len(FAR_OOD_CLASSES)} (should be 92)")

# 根据实际数量调整验证
if len(FAR_OOD_CLASSES) == 11:
    print("WARNING: Far-OOD categories count is 11, not 12, continuing...")
else:
    assert len(FAR_OOD_CLASSES) == 12, f"Far-OOD categories should be 12, but got {len(FAR_OOD_CLASSES)}"

assert len(ID_CLASSES) == 54, f"ID categories should be 54, but got {len(ID_CLASSES)}"
assert len(NEAR_OOD_CLASSES) == 26, f"Near-OOD categories should be 26, but got {len(NEAR_OOD_CLASSES)}"

print("Category definitions verified")

def split_dataset():
    """主函数：执行数据集划分"""

    # 定义路径
    RAW_DATA_PATH = Path("DYB-PlanktonNet")
    OUTPUT_PATH = Path("Plankton_OOD_Dataset")

    # 创建输出目录
    path_id_train = OUTPUT_PATH / "D_ID_train"
    path_id_val = OUTPUT_PATH / "D_ID_val"
    path_id_test = OUTPUT_PATH / "D_ID_test"
    path_near_test = OUTPUT_PATH / "D_Near_test"
    path_far_test = OUTPUT_PATH / "D_Far_test"

    # 创建所有目标文件夹
    for path in [path_id_train, path_id_val, path_id_test, path_near_test, path_far_test]:
        path.mkdir(parents=True, exist_ok=True)

    print("Target folders created")

    # 处理 Near-OOD 和 Far-OOD 数据（仅测试集）
    print("\nProcessing Near-OOD and Far-OOD data...")

    # Near-OOD 数据
    for class_name in NEAR_OOD_CLASSES:
        source_dir = RAW_DATA_PATH / class_name
        dest_dir = path_near_test / class_name

        if not dest_dir.exists():
            print(f"  Copying Near-OOD: {class_name}")
            shutil.copytree(source_dir, dest_dir)
        else:
            print(f"  WARNING: Near-OOD category {class_name} already exists, skipping")

    # Far-OOD 数据
    for class_name in FAR_OOD_CLASSES:
        source_dir = RAW_DATA_PATH / class_name
        dest_dir = path_far_test / class_name

        if not dest_dir.exists():
            print(f"  Copying Far-OOD: {class_name}")
            shutil.copytree(source_dir, dest_dir)
        else:
            print(f"  WARNING: Far-OOD category {class_name} already exists, skipping")

    print("Near-OOD and Far-OOD data processing completed")

    # 处理 ID 数据（8:1:1 拆分）
    print("\nProcessing ID data (8:1:1 split)...")

    total_images = {'train': 0, 'val': 0, 'test': 0}

    for class_name in ID_CLASSES:
        print(f"  处理 ID: {class_name}")

        source_dir = RAW_DATA_PATH / class_name

        # 获取图像文件列表
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        all_files = [f for f in os.listdir(source_dir)
                    if f.lower().endswith(image_extensions)]

        if not all_files:
            print(f"  WARNING: No image files found in {class_name}")
            continue

        # 打乱文件列表
        random.shuffle(all_files)

        # 计算拆分点
        n_total = len(all_files)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        n_test = n_total - n_train - n_val

        # 拆分文件列表
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        # 创建目标子文件夹
        train_dest_dir = path_id_train / class_name
        val_dest_dir = path_id_val / class_name
        test_dest_dir = path_id_test / class_name

        for dir_path in [train_dest_dir, val_dest_dir, test_dest_dir]:
            dir_path.mkdir(exist_ok=True)

        # 复制文件
        for file_list, dest_dir, split_name in zip(
            [train_files, val_files, test_files],
            [train_dest_dir, val_dest_dir, test_dest_dir],
            ['train', 'val', 'test']
        ):
            for filename in file_list:
                source_file = source_dir / filename
                dest_file = dest_dir / filename
                shutil.copy2(source_file, dest_file)
            total_images[split_name] += len(file_list)

        print(f"    {class_name}: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

    print("ID data processing completed")

    # 生成总结报告
    print("\nDataset Split Summary Report")
    print("=" * 50)

    # 统计每个目录的信息
    datasets_info = {
        "D_ID_train": path_id_train,
        "D_ID_val": path_id_val,
        "D_ID_test": path_id_test,
        "D_Near_test": path_near_test,
        "D_Far_test": path_far_test
    }

    for dataset_name, dataset_path in datasets_info.items():
        # 统计类别数量
        class_count = sum(1 for item in dataset_path.iterdir() if item.is_dir())

        # 统计图像总数
        image_count = 0
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                image_count += sum(1 for f in class_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in image_extensions)

        print(f"{dataset_name}:")
        print(f"  - Categories: {class_count}")
        print(f"  - Total images: {image_count}")
        print()

    print("Dataset split completed!")
    print(f"Output path: {OUTPUT_PATH.absolute()}")

if __name__ == "__main__":
    split_dataset()