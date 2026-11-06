
import os
import time
import statistics

import torch
import resnet
import utils

from thop import profile
from prune_model import prune_resnet34
from load_pruned import load_pruned_resnet34

# ==========================================
# 1. 实验配置
# ==========================================

DEVICE = torch.device("cpu")

KEEP_RATIO = 0.50
BATCH_SIZE = 64

WARMUP = 20
REPEAT = 100
CPU_THREADS = 4

torch.set_num_threads(CPU_THREADS)

print("=" * 50)
print("ResNet34 结构化剪枝性能测试")
print("=" * 50)

print("Device:", DEVICE)
print("CPU Threads:", torch.get_num_threads())
print("Keep Ratio:", KEEP_RATIO)


# ==========================================
# 2. 加载 CIFAR-10 数据集
# ==========================================

num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=BATCH_SIZE
)

print("\nCIFAR-10 加载完成")
print("类别数量:", num_classes)
print("测试图片数量:", len(test_loader.dataset))


# ==========================================
# 3. 创建并加载原始 ResNet34
# ==========================================

model = resnet.resnet34(
    num_classes=num_classes
)

checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["state_dict"])

model = model.to(DEVICE)
model.eval()

print("\n原始 ResNet34 加载成功")


# ==========================================
# 4. 执行结构化剪枝
# ==========================================

pruned_model = prune_resnet34(
    model,
    keep_ratio=KEEP_RATIO
)

pruned_model = pruned_model.to(DEVICE)
pruned_model.eval()

# 确保生成的是不同模型对象
assert pruned_model is not model

print("\n剪枝模型构建完成")


# ==========================================
# 5. 检查剪枝后的网络结构
# ==========================================

print("\n" + "=" * 50)
print("网络结构检查")
print("=" * 50)

print(
    "Original Conv1:",
    model.layer2[1].conv1.weight.shape
)

print(
    "Pruned Conv1:",
    pruned_model.layer2[1].conv1.weight.shape
)

print(
    "Original Conv2:",
    model.layer2[1].conv2.weight.shape
)

print(
    "Pruned Conv2:",
    pruned_model.layer2[1].conv2.weight.shape
)

# 检查完整模型的输出 Shape
dummy_input = torch.randn(
    1, 3, 32, 32,
    device=DEVICE
)

with torch.inference_mode():
    original_output = model(dummy_input)
    pruned_output = pruned_model(dummy_input)

assert original_output.shape == (1, 10)
assert pruned_output.shape == (1, 10)

print("Original Output Shape:", original_output.shape)
print("Pruned Output Shape:", pruned_output.shape)


# ==========================================
# 6. 统计 Parameters
# ==========================================

def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


original_params = count_params(model)
pruned_params = count_params(pruned_model)

assert pruned_params < original_params

params_reduction = (
    1 - pruned_params / original_params
) * 100

print("\n" + "=" * 50)
print("参数量统计")
print("=" * 50)

print(f"Original Params: {original_params:,}")
print(f"Pruned Params: {pruned_params:,}")
print(f"Params Reduction: {params_reduction:.2f}%")


# ==========================================
# 7. 统计 MACs
# ==========================================

with torch.inference_mode():

    original_macs, _ = profile(
        model,
        inputs=(dummy_input,),
        verbose=False
    )

    pruned_macs, _ = profile(
        pruned_model,
        inputs=(dummy_input,),
        verbose=False
    )


macs_reduction = (
    1 - pruned_macs / original_macs
) * 100

print("\n" + "=" * 50)
print("MACs 统计")
print("=" * 50)

print(f"Original MACs: {original_macs / 1e9:.4f} G")
print(f"Pruned MACs: {pruned_macs / 1e9:.4f} G")
print(f"MACs Reduction: {macs_reduction:.2f}%")


# ==========================================
# 8. CPU 推理延迟测试函数
# ==========================================

def measure_latency(
    model,
    input_tensor,
    warmup=20,
    repeat=100
):

    model.eval()
    times = []

    with torch.inference_mode():

        # 预热
        for _ in range(warmup):
            model(input_tensor)

        # 正式计时
        for _ in range(repeat):

            start = time.perf_counter()

            model(input_tensor)

            end = time.perf_counter()

            times.append(
                (end - start) * 1000
            )

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times)
    }


# ==========================================
# 9. 测量原始与剪枝模型 CPU Latency
# ==========================================

print("\n" + "=" * 50)
print("CPU 推理延迟测试")
print("=" * 50)

original_latency = measure_latency(
    model,
    dummy_input,
    warmup=WARMUP,
    repeat=REPEAT
)

pruned_latency = measure_latency(
    pruned_model,
    dummy_input,
    warmup=WARMUP,
    repeat=REPEAT
)

speedup = (
    original_latency["median"]
    / pruned_latency["median"]
)

print(
    f"Original Median Latency: "
    f"{original_latency['median']:.4f} ms"
)

print(
    f"Pruned Median Latency: "
    f"{pruned_latency['median']:.4f} ms"
)

print(f"Speedup: {speedup:.3f}x")


# ==========================================
# 10. Top-1 Accuracy 测试函数
# ==========================================

def evaluate_accuracy(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total * 100


# ==========================================
# 11. 比较分类准确率
# ==========================================

print("\n" + "=" * 50)
print("CIFAR-10 Top-1 Accuracy")
print("=" * 50)

original_acc = evaluate_accuracy(
    model,
    test_loader
)

pruned_acc = evaluate_accuracy(
    pruned_model,
    test_loader
)

accuracy_drop = original_acc - pruned_acc

print(f"Original Accuracy: {original_acc:.2f}%")
print(f"Pruned Accuracy: {pruned_acc:.2f}%")
print(f"Accuracy Drop: {accuracy_drop:.2f} pp")


# ==========================================
# 12. 保存剪枝后模型（新格式）
# ==========================================

def save_pruned_model(pruned_model, keep_ratio, save_path):

    # 1. 记录各个 BasicBlock 的内部通道数
    channel_config = {}

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(pruned_model, layer_name)

        for i, block in enumerate(layer):
            key = f"{layer_name}.{i}"
            channel_config[key] = block.conv1.out_channels
    # 清理 THOP 添加的统计 Buffer
    for module in pruned_model.modules():
        module._buffers.pop("total_ops", None)
        module._buffers.pop("total_params", None)
    # 2. 组装需要保存的所有信息
    checkpoint = {
        "model_type": "cifar_resnet34",
        "num_classes": pruned_model.linear.out_features,
        "keep_ratio": keep_ratio,
        "channel_config": channel_config,
        "state_dict": pruned_model.state_dict()
    }

    # 3. 保存到磁盘
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    saved = torch.load(
    save_path,
    map_location="cpu",
    weights_only=True
)

    print("Checkpoint 字段:", list(saved.keys()))

    assert saved["model_type"] == "cifar_resnet34"
    assert "channel_config" in saved
    assert "state_dict" in saved

save_path = (
    f"outputs/resnet34_pruned_{int(KEEP_RATIO * 100)}_v2.pth"
)

save_pruned_model(
    pruned_model,
    KEEP_RATIO,
    save_path
)

file_size_mb = os.path.getsize(save_path) / 1e6

print("\n模型保存成功！")
print("保存路径:", save_path)
print(f"Pruned Model File Size: {file_size_mb:.2f} MB")




# ==========================================
# 12.1 验证剪枝模型是否正确恢复
# ==========================================

print("\n开始验证模型保存与加载...")

# 重新从文件中加载剪枝模型
loaded_model = load_pruned_resnet34(save_path)

pruned_model.eval()
loaded_model.eval()

# 使用同一个输入比较
test_x = torch.randn(4, 3, 32, 32)

with torch.inference_mode():
    output_before = pruned_model(test_x)
    output_after = loaded_model(test_x)

# 计算两者的最大绝对误差
max_error = (
    output_before - output_after
).abs().max().item()

print(f"保存前后最大误差: {max_error:.10f}")

# 自动检查两者是否接近
torch.testing.assert_close(
    output_before,
    output_after,
    rtol=1e-5,
    atol=1e-6
)

print("模型保存与加载验证通过！")


# ==========================================
# 13. 最终实验结果汇总
# ==========================================

print("\n" + "=" * 50)
print("最终实验结果")
print("=" * 50)

print(f"Keep Ratio: {KEEP_RATIO:.2f}")
print(f"Original Params: {original_params / 1e6:.4f} M")
print(f"Pruned Params: {pruned_params / 1e6:.4f} M")
print(f"Params Reduction: {params_reduction:.2f}%")

print(f"Original MACs: {original_macs / 1e9:.4f} G")
print(f"Pruned MACs: {pruned_macs / 1e9:.4f} G")
print(f"MACs Reduction: {macs_reduction:.2f}%")

print(f"Original Accuracy: {original_acc:.2f}%")
print(f"Pruned Accuracy: {pruned_acc:.2f}%")
print(f"Accuracy Drop: {accuracy_drop:.2f} pp")

print(
    f"Original Median Latency: "
    f"{original_latency['median']:.4f} ms"
)

print(
    f"Pruned Median Latency: "
    f"{pruned_latency['median']:.4f} ms"
)

print(f"Speedup: {speedup:.3f}x")
print(f"Pruned Model File Size: {file_size_mb:.2f} MB")
