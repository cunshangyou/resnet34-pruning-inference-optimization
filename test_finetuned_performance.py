
import copy
import time
import statistics

import torch
import resnet

from thop import profile
from load_pruned import load_pruned_resnet34


# ==========================================
# 1. 实验配置
# ==========================================

DEVICE = torch.device("cpu")
CPU_THREADS = 4

WARMUP = 20
REPEAT = 100

torch.set_num_threads(CPU_THREADS)

print("=" * 55)
print("Fine-tuning 后模型性能对比实验")
print("=" * 55)

print("Device:", DEVICE)
print("CPU Threads:", torch.get_num_threads())


# ==========================================
# 2. 加载三个模型
# ==========================================

# 模型 A：原始 ResNet34
original_model = resnet.resnet34(num_classes=10)

checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

original_model.load_state_dict(checkpoint["state_dict"])
original_model.eval()


# 模型 B：50% 剪枝，未微调
pruned_model = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_v2.pth"
)


# 模型 C：50% 剪枝，累计微调 8 Epoch
finetuned_model = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_finetuned_stage2.pth"
)


models = {
    "Original": original_model,
    "Pruned": pruned_model,
    "Fine-tuned": finetuned_model
}

for model in models.values():
    model.to(DEVICE)
    model.eval()

print("\n三个模型均加载成功！")


# ==========================================
# 3. 统计 Parameters
# ==========================================

def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


print("\n" + "=" * 55)
print("Parameters 统计")
print("=" * 55)

param_results = {}

for name, model in models.items():

    params = count_params(model)
    param_results[name] = params

    fp32_size = params * 4 / 1e6

    print(f"{name}:")
    print(f"  Parameters: {params:,}")
    print(f"  Parameters: {params / 1e6:.4f} M")
    print(f"  FP32 Parameter Size: {fp32_size:.2f} MB")


# 验证微调没有改变剪枝模型结构
assert param_results["Pruned"] == param_results["Fine-tuned"]


# ==========================================
# 4. 统计 MACs
# ==========================================

dummy_input = torch.randn(
    1, 3, 32, 32,
    device=DEVICE
)

print("\n" + "=" * 55)
print("MACs 统计")
print("=" * 55)

macs_results = {}

for name, model in models.items():

    # 使用副本进行 THOP 统计，避免污染实际模型
    model_copy = copy.deepcopy(model)

    with torch.inference_mode():
        macs, _ = profile(
            model_copy,
            inputs=(dummy_input,),
            verbose=False
        )

    macs_results[name] = macs

    print(f"{name}: {macs / 1e9:.4f} G")


# 微调前后 MACs 应该完全相同
assert abs(
    macs_results["Pruned"] -
    macs_results["Fine-tuned"]
) < 1

print("\n微调前后 MACs 一致性检查通过！")


# ==========================================
# 5. 定义 CPU Latency 测试函数
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

        # 正式测试
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
# 6. 测量三个模型的 CPU Latency
# ==========================================

print("\n" + "=" * 55)
print("CPU Latency 测试")
print("=" * 55)

latency_results = {
    name: [] for name in models
}

# 每个模型先分别预热
for model in models.values():
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(dummy_input)


# 交替测试三个模型，减少测试顺序造成的偏差
model_names = list(models.keys())

for i in range(REPEAT):

    current_order = (
        model_names if i % 2 == 0
        else list(reversed(model_names))
    )

    for name in current_order:

        model = models[name]

        with torch.inference_mode():

            start = time.perf_counter()

            model(dummy_input)

            end = time.perf_counter()

        latency_results[name].append(
            (end - start) * 1000
        )


# 汇总延迟
for name in models:

    times = latency_results[name]

    print(f"\n{name}:")
    print(f"  Mean: {statistics.mean(times):.4f} ms")
    print(f"  Median: {statistics.median(times):.4f} ms")
    print(f"  Min: {min(times):.4f} ms")
    print(f"  Max: {max(times):.4f} ms")


# ==========================================
# 7. 计算加速比
# ==========================================

original_latency = statistics.median(
    latency_results["Original"]
)

pruned_latency = statistics.median(
    latency_results["Pruned"]
)

finetuned_latency = statistics.median(
    latency_results["Fine-tuned"]
)

pruned_speedup = original_latency / pruned_latency

finetuned_speedup = original_latency / finetuned_latency

latency_change = (
    finetuned_latency - pruned_latency
) / pruned_latency * 100


# ==========================================
# 8. 最终结果汇总
# ==========================================

print("\n" + "=" * 55)
print("最终实验结果")
print("=" * 55)

for name in models:

    print(f"\n{name}:")
    print(f"Parameters: {param_results[name] / 1e6:.4f} M")
    print(f"MACs: {macs_results[name] / 1e9:.4f} G")
    print(
        f"Median Latency: "
        f"{statistics.median(latency_results[name]):.4f} ms"
    )

print("\n" + "=" * 55)
print(f"Pruned Speedup: {pruned_speedup:.3f}x")
print(f"Fine-tuned Speedup: {finetuned_speedup:.3f}x")
print(f"Fine-tuning Latency Change: {latency_change:.2f}%")
print("=" * 55)
