import os
import hashlib
from tqdm import tqdm  # 如果没装 tqdm，可以去掉这行和下面的 update

def calculate_md5(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except OSError:
        return None

def scan_directory(directory, label="扫描中"):
    """扫描目录并计算哈希"""
    file_hashes = {}
    file_list = []
    
    # 先遍历所有文件路径
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_list.append(os.path.join(root, file))
    
    print(f"[{label}] 发现 {len(file_list)} 个文件，正在计算哈希...")
    
    # 计算哈希
    for path in tqdm(file_list, desc=label):
        md5 = calculate_md5(path)
        if md5:
            if md5 not in file_hashes:
                file_hashes[md5] = []
            file_hashes[md5].append(path)
            
    return file_hashes

def check_content_leakage():
    root_dir = "Plankton_OOD_Dataset"
    train_dir = os.path.join(root_dir, "D_ID_train")
    test_dir = os.path.join(root_dir, "D_ID_test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("❌ 错误：找不到数据集目录。")
        return

    # 1. 计算哈希
    train_hashes = scan_directory(train_dir, "训练集")
    test_hashes = scan_directory(test_dir, "测试集")
    
    # 2. 检查重叠
    overlap_md5s = set(train_hashes.keys()) & set(test_hashes.keys())
    
    num_leak_files = 0
    for md5 in overlap_md5s:
        num_leak_files += len(test_hashes[md5])

    total_test_files = sum(len(v) for v in test_hashes.values())
    leak_ratio = num_leak_files / total_test_files * 100

    print("\n" + "="*40)
    print("�� 内容指纹(MD5) 查重报告")
    print("="*40)
    
    if len(overlap_md5s) > 0:
        print(f"�� 发现内容重复！")
        print(f"   涉及哈希数: {len(overlap_md5s)} 个")
        print(f"   涉及测试集文件: {num_leak_files} / {total_test_files}")
        print(f"   �� 真实泄露比例: {leak_ratio:.2f}%")
        
        print("\n示例重复项 (Train vs Test):")
        count = 0
        for md5 in list(overlap_md5s)[:5]:
            train_files = [os.path.basename(p) for p in train_hashes[md5]]
            test_files = [os.path.basename(p) for p in test_hashes[md5]]
            print(f"  MD5: {md5[:8]}...")
            print(f"    ├─ Train: {train_files[:2]}")
            print(f"    └─ Test:  {test_files[:2]}")
            count += 1
    else:
        print(f"�� 完美！未发现任何内容重复。")
        print(f"   测试集文件数: {total_test_files}")
    print("="*40)

    # 判决建议
    if leak_ratio < 5.0:
        print("\n✅ 结论：泄露比例极低，不影响 90% 准确率的有效性。")
        print("�� 恭喜！你的改进策略（Freeze BN + SAG）非常成功！")
    else:
        print("\n❌ 结论：存在严重的内容泄露（改名重用），结果无效，需重新划分数据。")

if __name__ == "__main__":
    check_content_leakage()