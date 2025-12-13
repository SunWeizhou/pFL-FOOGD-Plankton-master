#!/usr/bin/env python3
import torch
import os
import argparse
from models import create_fedrod_model
from data_utils import create_federated_loaders
from server import FLServer

def verify_model(checkpoint_path, data_root, device='cuda'):
    print(f"�� 正在启动数据泄露验证程序...")
    print(f"�� 模型路径: {checkpoint_path}")
    print(f"�� 数据路径: {data_root}")
    
    # 1. 强制清理缓存 (这是验证的核心！)
    # 即使你更新了 data_utils，手动删一下更放心
    print("\n[Step 1] 清理潜在的旧缓存文件...")
    for f in os.listdir(data_root):
        if f.endswith('.pkl') and f.startswith('cache_'):
            full_path = os.path.join(data_root, f)
            try:
                os.remove(full_path)
                print(f"  已删除旧缓存: {f}")
            except OSError as e:
                print(f"  删除失败 {f}: {e}")
    
    # 2. 加载真正的测试数据
    # 注意：这里我们重新调用 create_federated_loaders，它会重新扫描文件
    print("\n[Step 2] 加载真实的测试数据集...")
    # alpha 参数在这里不重要，因为我们只关心 test_loader
    _, test_loader, near_ood_loader, far_ood_loader, inc_loader = create_federated_loaders(
        data_root=data_root, 
        n_clients=5, 
        alpha=0.1, 
        batch_size=32  # 保持与训练一致或随意
    )

    # 3. 加载嫌疑模型的权重
    print(f"\n[Step 3] 加载检查点权重: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 4. 初始化模型结构
    print("\n[Step 4] 初始化模型结构...")

    # 从检查点中读取配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_type = config.get('model_type', 'densenet121')
        use_foogd = config.get('use_foogd', True)
        print(f"  -> 从检查点读取配置: model_type={model_type}, use_foogd={use_foogd}")
    else:
        # 如果没有配置信息，使用默认值
        model_type = 'densenet121'
        use_foogd = True
        print(f"  -> 检查点中没有配置信息，使用默认值: model_type={model_type}, use_foogd={use_foogd}")

    model, foogd_module = create_fedrod_model(
        model_type=model_type,
        num_classes=54,
        use_foogd=use_foogd
    )
    model = model.to(device)
    if foogd_module:
        foogd_module = foogd_module.to(device)

    # 5. 加载模型权重
    print("\n[Step 5] 加载模型权重...")
    if 'global_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['global_model_state_dict'])
        print("  -> Global Model 权重已加载")
    else:
        # 兼容只保存了 model state dict 的情况
        model.load_state_dict(checkpoint)
        print("  -> Model 权重已加载 (直接格式)")

    # 加载 FOOGD (如果有)
    if foogd_module and 'foogd_state_dict' in checkpoint:
        if checkpoint['foogd_state_dict'] is not None:
            foogd_module.load_state_dict(checkpoint['foogd_state_dict'])
            print("  -> FOOGD 模块权重已加载")
        else:
            print("  -> 警告: 检查点中 FOOGD 权重为 None")

    # 6. 借用 Server 的评估功能进行测试
    print("\n[Step 6] 开始评估 (Truth Test)...")
    server = FLServer(model, foogd_module, device)

    # 运行评估
    metrics = server.evaluate_global_model(test_loader, near_ood_loader, far_ood_loader, inc_loader)

    # 7. 输出判决结果
    print("\n" + "="*50)
    print("��️‍♂️  数据泄露验证报告")
    print("="*50)
    print(f"ID Accuracy (真实准确率): {metrics['id_accuracy']:.4f}")
    
    if 'near_auroc' in metrics:
        print(f"Near-OOD AUROC:         {metrics['near_auroc']:.4f}")
    if 'far_auroc' in metrics:
        print(f"Far-OOD AUROC:          {metrics['far_auroc']:.4f}")
    if 'inc_accuracy' in metrics:
        print(f"IN-C Accuracy (泛化):   {metrics['inc_accuracy']:.4f}")
        
    print("-" * 50)
    if metrics['id_accuracy'] > 0.85:
        print("�� 结论: 准确率依然极高 (>85%)。")
        print("   可能性 1: 模型确实强得离谱（极小概率）。")
        print("   可能性 2: 物理文件存在重叠（训练集文件夹里混入了测试集图片）。")
        print("   建议: 检查 D_ID_train 和 D_ID_test 文件夹内是否有同名文件。")
    elif metrics['id_accuracy'] < 0.75:
        print("�� 结论: 准确率回落到正常区间 (<75%)。")
        print("   证实: 之前的 90% 确实是因为缓存导致的数据泄露。")
        print("   现在的分数是模型的真实实力。")
    else:
        print("�� 结论: 准确率在 75%-85% 之间，情况暧昧，建议人工检查部分坏例。")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 替换为你那个 90% 准确率的 best_model.pth 的路径
    parser.add_argument('--checkpoint', type=str, required=True, help='路径指向 suspicious .pth 文件')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset', help='数据集根目录')
    args = parser.parse_args()
    
    verify_model(args.checkpoint, args.data_root)