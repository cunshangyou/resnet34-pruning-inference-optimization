import torch
import utils

from load_pruned import load_pruned_resnet34

# 1. 加载第二阶段微调模型
model = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_finetuned_stage2.pth"
)

model.eval()

# 2. 加载 CIFAR-10 测试集
num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=64
)

# 3. 计算准确率
correct = 0
total = 0

with torch.inference_mode():

    for images, labels in test_loader:

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print("正确数量:", correct)
print("测试总数量:", total)
print(f"Stage2 Accuracy: {accuracy:.2f}%")

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)