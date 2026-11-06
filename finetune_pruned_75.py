import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from load_pruned import load_pruned_resnet34


# ==========================================
# 1. 实验配置
# ==========================================

DEVICE = torch.device("cpu")

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CPU_THREADS = 4

torch.set_num_threads(CPU_THREADS)

SOURCE_PATH = "outputs/resnet34_pruned_75_v2.pth"

SAVE_PATH = "outputs/resnet34_pruned_75_finetuned.pth"


# ==========================================
# 2. 加载剪枝模型
# ==========================================

model = load_pruned_resnet34(
    SOURCE_PATH
).to(DEVICE)

print(
    "模型参数量:",
    sum(p.numel() for p in model.parameters())
)

print(
    "Layer2 Conv1:",
    model.layer2[1].conv1.weight.shape
)

# ==========================================
# 3. 加载 CIFAR-10
# ==========================================

num_classes, train_loader, test_loader = utils.get_dataset(
    "cifar10",
    "datasets/",
    batch_size=BATCH_SIZE
)

print("训练集数量:", len(train_loader.dataset))
print("测试集数量:", len(test_loader.dataset))


# ==========================================
# 4. 创建 Loss 和 Optimizer
# ==========================================

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4
)

# ==========================================
# 5. 训练一个 Epoch
# ==========================================

def train_one_epoch(model, train_loader, optimizer, loss_fn):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start_time = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(
        train_loader, start=1
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向传播
        outputs = model(images)

        loss = loss_fn(outputs, labels)

        # 反向传播
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

        # 每 100 个 Batch 输出一次
        if batch_idx % 100 == 0 or batch_idx == len(train_loader):

            current_loss = total_loss / total_samples

            current_acc = (
                total_correct / total_samples * 100
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {current_loss:.4f} "
                f"Accuracy: {current_acc:.2f}% "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

    avg_loss = total_loss / total_samples

    train_acc = total_correct / total_samples * 100

    return avg_loss, train_acc

# ==========================================
# 6. 准备训练 Checkpoint
# ==========================================

start_epoch = 0

checkpoint = torch.load(
    SOURCE_PATH,
    map_location="cpu",
    weights_only=True
)

if os.path.exists(SAVE_PATH):

    resume_checkpoint = torch.load(
        SAVE_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        resume_checkpoint["state_dict"]
    )

    if "optimizer_state_dict" in resume_checkpoint:
        optimizer.load_state_dict(
            resume_checkpoint["optimizer_state_dict"]
        )

    start_epoch = resume_checkpoint["completed_epoch"]

    checkpoint = resume_checkpoint

    print("发现已有训练记录，恢复训练！")
    print("已经完成 Epoch:", start_epoch)



# ==========================================
# 7. 开始 Fine-tuning
# ==========================================

for epoch in range(start_epoch, EPOCHS):

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

    # 保存当前进度
    checkpoint["state_dict"] = model.state_dict()

    checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    checkpoint["completed_epoch"] = epoch + 1

    checkpoint["finetune_epochs"] = epoch + 1

    checkpoint["finetune_lr"] = LEARNING_RATE

    torch.save(checkpoint, SAVE_PATH)

    print("Checkpoint 已保存:", SAVE_PATH)



# ==========================================
# 8. 测试微调后的准确率
# ==========================================

model.eval()

correct = 0
total = 0

with torch.inference_mode():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

accuracy = correct / total * 100

print("\n" + "=" * 45)
print("75% 剪枝模型 Fine-tuning 结果")
print("=" * 45)

print("正确数量:", correct)
print("测试总数量:", total)

print(f"Fine-tuned Accuracy: {accuracy:.2f}%")

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)