
import torch
import utils

from load_pruned import load_pruned_resnet34

DEVICE = torch.device("cpu")

# 1. 加载微调后的剪枝模型
model = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_finetuned.pth"
).to(DEVICE)

model.eval()

# 2. 加载 CIFAR-10 测试集
num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=64
)

# 3. 测试准确率
correct = 0
total = 0

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100

print("微调后预测正确数量:", correct)
print("测试总数量:", total)
print(f"Fine-tuned Accuracy: {accuracy:.2f}%")
