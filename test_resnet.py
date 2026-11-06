import torch
import resnet

# 1. 创建网络结构
model = resnet.resnet34(num_classes=10)

# 2. 读取预训练权重
checkpoint = torch.load(
    "pretrained/cifar10_resnet34.pth",
    map_location="cpu",
    weights_only=True
)

# 3. 加载参数
model.load_state_dict(checkpoint["state_dict"])

# 4. 切换到推理模式
model.eval()

# 5. 使用随机数据验证网络结构
x = torch.randn(1, 3, 32, 32)

with torch.inference_mode():
    output = model(x)

print("模型加载成功！")
print("输入 Shape:", x.shape)
print("输出 Shape:", output.shape)