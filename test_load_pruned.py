
import torch

checkpoint = torch.load(
    "outputs/resnet34_pruned_75_v2.pth",
    map_location="cpu",
    weights_only=True
)

print("Checkpoint 包含的字段：")
print(list(checkpoint.keys())[:15])

from load_pruned import load_pruned_resnet34


# 1. 直接读取已经保存的剪枝模型
model = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_v2.pth"
)

# 2. 检查参数量
params = sum(p.numel() for p in model.parameters())

print("加载后参数量:", params)

# 3. 检查某个卷积层 Shape
print(
    "Layer2 Conv1 Shape:",
    model.layer2[1].conv1.weight.shape
)

# 4. 测试完整网络前向传播
x = torch.randn(1, 3, 32, 32)

with torch.inference_mode():
    output = model(x)

print("输出 Shape:", output.shape)

# 5. 自动验证
assert params == 16008682
assert output.shape == (1, 10)

print("剪枝模型独立加载成功！")
