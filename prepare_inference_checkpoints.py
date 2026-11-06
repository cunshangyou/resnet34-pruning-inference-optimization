
import os
import torch


# ==========================================
# 1. 原始模型文件
# ==========================================

model_paths = {
    "Pruned_75":
        "outputs/resnet34_pruned_75_v2.pth",

    "Finetuned_75":
        "outputs/resnet34_pruned_75_finetuned.pth",

    "Pruned_50":
        "outputs/resnet34_pruned_50_v2.pth",

    "Finetuned_50":
        "outputs/resnet34_pruned_50_finetuned_stage2.pth"
}


# ==========================================
# 2. 创建推理专用目录
# ==========================================

save_dir = "outputs/inference_only"

os.makedirs(save_dir, exist_ok=True)


# ==========================================
# 3. 提取纯推理信息
# ==========================================

for name, path in model_paths.items():

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    # 只提取模型参数
    state_dict = checkpoint["state_dict"]

    # 清理 THOP 统计信息
    clean_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key.split(".")[-1] not in (
            "total_ops",
            "total_params"
        )
    }

    # 构造推理专用 Checkpoint
    inference_checkpoint = {
        "model_type": checkpoint["model_type"],
        "num_classes": checkpoint["num_classes"],
        "keep_ratio": checkpoint["keep_ratio"],
        "channel_config": checkpoint["channel_config"],
        "state_dict": clean_state_dict
    }

    save_path = os.path.join(
        save_dir,
        f"{name}.pth"
    )

    torch.save(
        inference_checkpoint,
        save_path
    )

    size_mb = os.path.getsize(save_path) / 1e6

    print(f"{name}: {size_mb:.2f} MB")

print("\n所有推理专用模型保存完成！")
