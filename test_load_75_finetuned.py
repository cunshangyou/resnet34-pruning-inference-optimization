import torch

from load_pruned import load_pruned_resnet34

model = load_pruned_resnet34(
    "outputs/resnet34_pruned_75_finetuned.pth"
)

model.eval()

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

x = torch.randn(1, 3, 32, 32)

with torch.inference_mode():
    output = model(x)

print("Output Shape:", output.shape)