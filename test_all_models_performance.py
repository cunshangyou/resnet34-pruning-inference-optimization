
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
print("ResNet34 多组剪枝模型性能对比")
print("=" * 55)

print("Device:", DEVICE)
print("CPU Threads:", torch.get_num_threads())


# ==========================================
# 2. 加载五个模型
# ==========================================

# A. 原始 ResNet34
original_model = resnet.resnet34(num_classes=10)

checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

original_model.load_state_dict(checkpoint["state_dict"])
# 释放原始 Checkpoint，避免同时保留两份模型权重
del checkpoint

import gc
gc.collect()

# B. 75% 剪枝，未微调
pruned_75 = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_v2.pth"
)


# C. 75% 剪枝，微调 3 Epoch
finetuned_75 = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_finetuned.pth"
)


# D. 50% 剪枝，未微调
pruned_50 = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_v2.pth"
)


# E. 50% 剪枝，微调 8 Epoch
finetuned_50 = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_finetuned_stage2.pth"
)


models = {
    "Original": original_model,
    "Pruned_75": pruned_75,
    "Finetuned_75": finetuned_75,
    "Pruned_50": pruned_50,
    "Finetuned_50": finetuned_50
}

for model in models.values():
    model.to(DEVICE)
    model.eval()

print("\n五个模型全部加载成功！")


# ==========================================
# 3. 统计 Parameters
# ==========================================

def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


param_results = {}

print("\n" + "=" * 55)
print("Parameters 统计")
print("=" * 55)

for name, model in models.items():

    params = count_params(model)

    param_results[name] = params

    print(
        f"{name}: "
        f"{params:,} "
        f"({params / 1e6:.4f} M)"
    )


# 检查微调前后参数量是否一致
assert param_results["Pruned_75"] == param_results["Finetuned_75"]

assert param_results["Pruned_50"] == param_results["Finetuned_50"]


# ==========================================
# 4. 统计 MACs
# ==========================================

dummy_input = torch.randn(
    1, 3, 32, 32,
    device=DEVICE
)

macs_results = {}

print("\n" + "=" * 55)
print("MACs 统计")
print("=" * 55)

for name, model in models.items():

    # 使用模型副本进行 THOP 统计
    model_copy = copy.deepcopy(model)

    with torch.inference_mode():

        macs, _ = profile(
            model_copy,
            inputs=(dummy_input,),
            verbose=False
        )

    macs_results[name] = macs

    print(f"{name}: {macs / 1e9:.4f} G")


# 微调前后 MACs 一致性检查
assert abs(
    macs_results["Pruned_75"] -
    macs_results["Finetuned_75"]
) < 1

assert abs(
    macs_results["Pruned_50"] -
    macs_results["Finetuned_50"]
) < 1

print("\n微调前后 MACs 一致性检查通过！")


# ==========================================
# 5. CPU Latency 测试函数
# ==========================================

def measure_latency(model, input_tensor, warmup=20, repeat=100):

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

    return times


# ==========================================
# 6. 测量五个模型的 CPU Latency
# ==========================================

print("\n" + "=" * 55)
print("CPU Latency 测试")
print("=" * 55)

latency_results = {
    name: [] for name in models
}

# 每个模型预热
for model in models.values():

    with torch.inference_mode():

        for _ in range(WARMUP):
            model(dummy_input)


# 交替测试，降低测试顺序带来的影响
model_names = list(models.keys())

for i in range(REPEAT):

    # 每轮轮换测试顺序
    current_order = (
        model_names[i % len(model_names):]
        + model_names[:i % len(model_names)]
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
    print(f"Mean: {statistics.mean(times):.4f} ms")
    print(f"Median: {statistics.median(times):.4f} ms")
    print(f"Min: {min(times):.4f} ms")
    print(f"Max: {max(times):.4f} ms")


# ==========================================
# 7. 计算加速比
# ==========================================

original_latency = statistics.median(
    latency_results["Original"]
)

print("\n" + "=" * 55)
print("加速比与延迟降低比例")
print("=" * 55)

for name in models:

    current_latency = statistics.median(
        latency_results[name]
    )

    speedup = original_latency / current_latency

    reduction = (
        1 - current_latency / original_latency
    ) * 100

    print(f"\n{name}:")
    print(f"Speedup: {speedup:.3f}x")
    print(f"Latency Reduction: {reduction:.2f}%")


# ==========================================
# 8. 最终结果汇总
# ==========================================

print("\n" + "=" * 65)
print("最终实验结果汇总")
print("=" * 65)

print(
    f"{'Model':<18}"
    f"{'Params(M)':<13}"
    f"{'MACs(G)':<12}"
    f"{'Latency(ms)':<15}"
    f"{'Speedup':<10}"
)

for name in models:

    params = param_results[name] / 1e6

    macs = macs_results[name] / 1e9

    latency = statistics.median(
        latency_results[name]
    )

    speedup = original_latency / latency

    print(
        f"{name:<18}"
        f"{params:<13.4f}"
        f"{macs:<12.4f}"
        f"{latency:<15.4f}"
        f"{speedup:<10.3f}"
    )

print("=" * 65)
print("全部测试完成！")
