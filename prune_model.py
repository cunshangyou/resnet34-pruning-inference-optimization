import torch
import torch.nn as nn
import copy

def pur_basicblock(block,keep_ratio=0.75):
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio 必须在 (0,1] 范围内")
    old_conv1 = block.conv1
    out_channels = old_conv1.out_channels
    keep_channels = max(
        1,
        int(out_channels*keep_ratio)
    )
    scores = (
        old_conv1.weight.detach()
        .abs()
        .sum(dim = (1,2,3))
    )
    keep_idx = torch.argsort(
        scores,
        descending=True
    )[:keep_channels]

    keep_idx = torch.sort(keep_idx).values
    print("原始通道数:", out_channels)
    print("保留通道数:", keep_channels)
    print("保留索引 Shape:", keep_idx.shape)

    new_conv1 = nn.Conv2d(
    in_channels=old_conv1.in_channels,
    out_channels=keep_channels,
    kernel_size=old_conv1.kernel_size,
    stride=old_conv1.stride,
    padding=old_conv1.padding,
    dilation=old_conv1.dilation,
    groups=old_conv1.groups,
    bias=old_conv1.bias is not None
).to(
    device=old_conv1.weight.device,
    dtype=old_conv1.weight.dtype
)
    with torch.no_grad():
        new_conv1.weight.copy_(
        old_conv1.weight[keep_idx]##第0维度是输出
        )

    old_conv2 = block.conv2

    new_conv2 = nn.Conv2d(
    in_channels=keep_channels,
    out_channels=old_conv2.out_channels,
    kernel_size=old_conv2.kernel_size,
    stride=old_conv2.stride,
    padding=old_conv2.padding,
    dilation=old_conv2.dilation,
    groups=old_conv2.groups,
    bias=old_conv2.bias is not None
).to(
    device=old_conv2.weight.device,
    dtype=old_conv2.weight.dtype
)

    with torch.no_grad():
        new_conv2.weight.copy_(
        old_conv2.weight[:, keep_idx, :, :]##第1维度是输入
        )


    old_bn1 = block.bn1
    new_bn1 = nn.BatchNorm2d(
    num_features=keep_channels,
    eps=old_bn1.eps,
    momentum=old_bn1.momentum
).to(
    device=old_bn1.weight.device,
    dtype=old_bn1.weight.dtype
)
    
    with torch.no_grad():
        new_bn1.weight.copy_(old_bn1.weight[keep_idx])
        new_bn1.bias.copy_(old_bn1.bias[keep_idx])

        new_bn1.running_mean.copy_(
            old_bn1.running_mean[keep_idx]
        )
        new_bn1.running_var.copy_(
            old_bn1.running_var[keep_idx]
        )

        new_bn1.num_batches_tracked.copy_(
            old_bn1.num_batches_tracked
        )


    pruned_block = copy.deepcopy(block)

    pruned_block.conv1 = new_conv1
    pruned_block.bn1 = new_bn1
    pruned_block.conv2 = new_conv2

    # 与原始 Block 的训练/推理模式保持一致
    pruned_block.train(block.training)

    return pruned_block,keep_idx


def prune_resnet34(model, keep_ratio=0.75):

    pruned_model = copy.deepcopy(model)

    for layer_name in [
        "layer1",
        "layer2",
        "layer3",
        "layer4"
    ]:
        layer = getattr(pruned_model, layer_name)

        for i, block in enumerate(layer):

            new_block, keep_idx = pur_basicblock(
                block,
                keep_ratio=keep_ratio
            )

            layer[i] = new_block

    return pruned_model

if __name__ == "__main__":
    import resnet

    model = resnet.resnet34(num_classes=10)
    model.eval()

    pruned_model = prune_resnet34(
        model,
        keep_ratio=0.75
    )

    pruned_model.eval()

    x = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        output = pruned_model(x)

    print("输出 Shape:", output.shape)



    original_params = sum(
        p.numel() for p in model.parameters()
    )

    pruned_params = sum(
        p.numel() for p in pruned_model.parameters()
    )

    print("原始参数量:", original_params)
    print("剪枝后参数量:", pruned_params)

    reduction = (
        1 - pruned_params / original_params
    ) * 100

    print(f"参数减少比例: {reduction:.2f}%")