import torch
import resnet
import utils

from prune_model import prune_resnet34

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

# 3. 剪枝 25% 内部通道
pruned_model = prune_resnet34(
    model,
    keep_ratio=0.75
)

pruned_model.eval()

num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=64
)

def evaluate(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


original_acc = evaluate(model, test_loader)
pruned_acc = evaluate(pruned_model, test_loader)

print(f"Original Accuracy: {original_acc:.2f}%")
print(f"Pruned Accuracy: {pruned_acc:.2f}%")
print(
    f"Accuracy Drop: {original_acc - pruned_acc:.2f} percentage points"
)





from thop import profile

x = torch.randn(1, 3, 32, 32)

pruned_model.eval()

with torch.inference_mode():
    macs, params = profile(
        pruned_model,
        inputs=(x,),
        verbose=False
    )

print(f"Pruned MACs: {macs / 1e9:.4f} G")
print(f"Pruned Parameters: {params / 1e6:.4f} M")

reduction = (1 - macs / 1.1635e9) * 100

print(f"MACs Reduction: {reduction:.2f}%")


warmup = 20
repeat = 100
batch_size = 1