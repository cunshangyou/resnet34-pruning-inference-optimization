
import os
import io
import torch


# ==========================================
# 1. 定义五个模型文件路径
# ==========================================

model_paths = {
    "Original": "pretrained/cifar10_resnet34.pth",

    "Pruned_75": "outputs/resnet34_pruned_75_v2.pth",

    "Finetuned_75": "outputs/resnet34_pruned_75_finetuned.pth",

    "Pruned_50": "outputs/resnet34_pruned_50_v2.pth",

    "Finetuned_50": "outputs/resnet34_pruned_50_finetuned_stage2.pth"
}


# ==========================================
# 2. 统一提取模型权重
# ==========================================

def get_state_dict(checkpoint):

    # 新格式 Checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    # 兼容直接保存 state_dict 的旧格式
    else:
        state_dict = checkpoint

    # 排除 THOP 临时添加的统计信息
    clean_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key not in ("total_ops", "total_params")
        and isinstance(value, torch.Tensor)
    }

    return clean_state_dict


# ==========================================
# 3. 测量文件大小
# ==========================================

results = {}

print("=" * 65)
print("ResNet34 五组模型文件大小对比")
print("=" * 65)

for name, path in model_paths.items():

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"模型文件不存在: {path}"
        )

    # 读取原始 Checkpoint
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # 提取纯模型权重
    state_dict = get_state_dict(checkpoint)

    # 计算实际 Checkpoint 文件大小
    checkpoint_size = os.path.getsize(path) / 1e6

    # 使用统一的 torch.save 格式序列化权重
    buffer = io.BytesIO()

    torch.save(state_dict, buffer)

    weight_size = buffer.getbuffer().nbytes / 1e6

    results[name] = {
        "checkpoint_size": checkpoint_size,
        "weight_size": weight_size
    }

    print(f"\n{name}:")
    print(f"  Checkpoint Size: {checkpoint_size:.2f} MB")
    print(f"  State Dict Size: {weight_size:.2f} MB")


# ==========================================
# 4. 计算权重存储压缩比例
# ==========================================

original_size = results["Original"]["weight_size"]

print("\n" + "=" * 65)
print("统一格式下的模型权重大小对比")
print("=" * 65)

for name, result in results.items():

    weight_size = result["weight_size"]

    reduction = (
        1 - weight_size / original_size
    ) * 100

    print(
        f"{name:<15}"
        f"Weight Size: {weight_size:>8.2f} MB    "
        f"Reduction: {reduction:>6.2f}%"
    )


# ==========================================
# 5. 检查微调前后的权重大小
# ==========================================

size_75_diff = abs(
    results["Pruned_75"]["weight_size"]
    - results["Finetuned_75"]["weight_size"]
)

size_50_diff = abs(
    results["Pruned_50"]["weight_size"]
    - results["Finetuned_50"]["weight_size"]
)

print("\n" + "=" * 65)
print("微调前后权重大小一致性检查")
print("=" * 65)

print(f"75% 模型大小差异: {size_75_diff:.6f} MB")
print(f"50% 模型大小差异: {size_50_diff:.6f} MB")

print("文件大小测试完成！")
