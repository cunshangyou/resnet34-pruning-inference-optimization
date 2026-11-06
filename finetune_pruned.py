import os
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from load_pruned import load_pruned_resnet34

# 实验配置
DEVICE = torch.device("cpu")
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.01

torch.set_num_threads(4)

# 加载已经剪枝的模型
model = load_pruned_resnet34(
    "outputs/resnet34_pruned_50_v2.pth"
).to(DEVICE)

# 加载 CIFAR-10
num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=BATCH_SIZE
)

print("模型参数量:", sum(p.numel() for p in model.parameters()))
print("训练集数量:", len(train_loader.dataset))
print("测试集数量:", len(test_loader.dataset))

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4
)


def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向传播
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples
    train_acc = total_correct / total_samples * 100

    return avg_loss, train_acc


for epoch in range(EPOCHS):

    avg_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}, "
        f"Train Acc: {train_acc:.2f}%"
    )

model.eval()

os.makedirs("outputs", exist_ok=True)

# 保留原始剪枝模型文件，不覆盖它
source_path = "outputs/resnet34_pruned_50_v2.pth"
save_path = "outputs/resnet34_pruned_50_finetuned.pth"

# 继承已有的结构配置及实验信息
checkpoint = torch.load(
    source_path,
    map_location="cpu",
    weights_only=True
)

checkpoint["state_dict"] = model.state_dict()
checkpoint["finetune_epochs"] = EPOCHS
checkpoint["finetune_lr"] = LEARNING_RATE

torch.save(checkpoint, save_path)

print("微调模型保存成功:", save_path)