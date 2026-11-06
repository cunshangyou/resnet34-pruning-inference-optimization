import torch
import time
import statistics#干统计数据的
import resnet

# 1. 创建模型
model = resnet.resnet34(num_classes=10)

# 2. 加载预训练权重
checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 3. 创建测试输入
x = torch.randn(1, 3, 32, 32)

warmup = 20
repeat = 100

times = []

with torch.inference_mode():

    # 4. 预热
    for _ in range(warmup):
        model(x)

    # 5. 正式计时
    for _ in range(repeat):

        start = time.perf_counter()

        model(x)

        end = time.perf_counter()

        times.append((end - start) * 1000)

# 6. 计算统计指标
mean_ms = statistics.mean(times)#算平均值
median_ms = statistics.median(times)#算中位数的

print(f"Mean Latency: {mean_ms:.4f} ms")
print(f"Median Latency: {median_ms:.4f} ms")
print(f"Min Latency: {min(times):.4f} ms")
print(f"Max Latency: {max(times):.4f} ms")