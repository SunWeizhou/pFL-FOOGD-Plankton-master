import os
import hashlib

def get_file_hashes(directory):
    """计算目录下所有文件的MD5哈希值"""
    file_hashes = {}
    print(f"正在扫描: {directory} ...")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
                path = os.path.join(root, file)
                # 计算简单哈希（为了速度，先用文件名+大小，如果需要更严谨可以用MD5）
                # 这里为了直观，我们直接检测【文件名】重叠，因为通常文件名是唯一的
                file_hashes[file] = path
    return file_hashes

def check_leakage():
    root_dir = "Plankton_OOD_Dataset"
    train_dir = os.path.join(root_dir, "D_ID_train")
    test_dir = os.path.join(root_dir, "D_ID_test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("找不到数据集目录，请确认路径。")
        return

    train_files = get_file_hashes(train_dir)
    test_files = get_file_hashes(test_dir)
    
    print(f"训练集文件数: {len(train_files)}")
    print(f"测试集文件数: {len(test_files)}")
    
    # 检查重叠
    overlap = set(train_files.keys()) & set(test_files.keys())
    
    print("\n" + "="*30)
    print(f"�� 发现重叠文件数: {len(overlap)}")
    print("="*30)
    
    if len(overlap) > 0:
        print(f"泄露比例: {len(overlap) / len(test_files) * 100:.2f}% 的测试集数据在训练集中出现过！")
        print("示例重叠文件:")
        for i, f in enumerate(list(overlap)[:5]):
            print(f"  - {f}")
    else:
        print("�� 未发现文件名重叠。")

if __name__ == "__main__":
    check_leakage()