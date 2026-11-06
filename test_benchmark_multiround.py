
import time
import statistics

import torch
import resnet

from load_pruned import load_pruned_resnet34


# ==========================================
# 1. 实验配置
# ==========================================

DEVICE = torch.device("cpu")

CPU_THREADS = 4

WARMUP = 20
REPEAT = 100
ROUNDS = 5

torch.set_num_threads(CPU_THREADS)


# ==========================================
# 2. 加载五个模型
# ==========================================

original_model = resnet.resnet34(num_classes=10)

checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

original_model.load_state_dict(
    checkpoint["state_dict"]
)

del checkpoint


pruned_75 = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_v2.pth"
)

finetuned_75 = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_finetuned.pth"
)

pruned_50 = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_v2.pth"
)

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


# ==========================================
# 3. 准备统一输入
# ==========================================

dummy_input = torch.randn(
    1, 3, 32, 32,
    device=DEVICE
)


# ==========================================
# 4. 初始化结果容器
# ==========================================

round_results = {
    name: [] for name in models
}


# ==========================================
# 5. 多轮 Benchmark
# ==========================================

print("=" * 65)
print("ResNet34 多轮 CPU Benchmark")
print("=" * 65)

print("CPU Threads:", torch.get_num_threads())
print("Warmup:", WARMUP)
print("Repeat:", REPEAT)
print("Rounds:", ROUNDS)


for round_idx in range(ROUNDS):

    print(f"\n开始第 {round_idx + 1}/{ROUNDS} 轮测试")

    # 每轮先对五个模型进行预热
    for model in models.values():

        with torch.inference_mode():

            for _ in range(WARMUP):
                model(dummy_input)

    # 每轮轮换模型测试顺序
    model_names = list(models.keys())

    offset = round_idx % len(model_names)

    current_order = (
        model_names[offset:]
        + model_names[:offset]
    )

    for name in current_order:

        model = models[name]

        times = []

        with torch.inference_mode():

            for _ in range(REPEAT):

                start = time.perf_counter()

                model(dummy_input)

                end = time.perf_counter()

                times.append(
                    (end - start) * 1000
                )

        median_ms = statistics.median(times)

        round_results[name].append(median_ms)

        print(
            f"{name}: {median_ms:.4f} ms"
        )


# ==========================================
# 6. 汇总五轮结果
# ==========================================

print("\n" + "=" * 80)
print("五轮 Benchmark 结果汇总")
print("=" * 80)

summary = {}

for name, values in round_results.items():

    median_ms = statistics.median(values)

    mean_ms = statistics.mean(values)

    min_ms = min(values)

    max_ms = max(values)

    summary[name] = {
        "median": median_ms,
        "mean": mean_ms,
        "min": min_ms,
        "max": max_ms
    }

    print(f"\n{name}:")
    print("每轮 Median:", [round(v, 4) for v in values])

    print(f"五轮中位数: {median_ms:.4f} ms")
    print(f"五轮平均值: {mean_ms:.4f} ms")
    print(f"最小值: {min_ms:.4f} ms")
    print(f"最大值: {max_ms:.4f} ms")


# ==========================================
# 7. 计算最终加速比
# ==========================================

original_latency = summary["Original"]["median"]

print("\n" + "=" * 80)
print("最终加速比")
print("=" * 80)

for name, result in summary.items():

    speedup = (
        original_latency / result["median"]
    )

    reduction = (
        1 - result["median"] / original_latency
    ) * 100

    print(f"\n{name}:")
    print(f"Speedup: {speedup:.3f}x")
    print(f"Latency Reduction: {reduction:.2f}%")


# ==========================================
# 8. 最终对照表
# ==========================================

print("\n" + "=" * 85)
print("最终多轮 Benchmark 结果")
print("=" * 85)

print(
    f"{'Model':<18}"
    f"{'Median(ms)':<15}"
    f"{'Mean(ms)':<15}"
    f"{'Min(ms)':<13}"
    f"{'Max(ms)':<13}"
    f"{'Speedup':<10}"
)

for name, result in summary.items():

    speedup = (
        original_latency / result["median"]
    )

    print(
        f"{name:<18}"
        f"{result['median']:<15.4f}"
        f"{result['mean']:<15.4f}"
        f"{result['min']:<13.4f}"
        f"{result['max']:<13.4f}"
        f"{speedup:<10.3f}"
    )

print("\n多轮 Benchmark 完成！")
