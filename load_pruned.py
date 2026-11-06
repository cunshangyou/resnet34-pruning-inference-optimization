import torch
import torch.nn as nn
import resnet


def rebuild_block(block, mid_channels):

    old_conv1 = block.conv1
    old_conv2 = block.conv2
    old_bn1 = block.bn1

    # 1. 创建新的 Conv1
    block.conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels,
        out_channels=mid_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        dilation=old_conv1.dilation,
        groups=old_conv1.groups,
        bias=old_conv1.bias is not None
    )

    # 2. 创建新的 BN1
    block.bn1 = nn.BatchNorm2d(
        num_features=mid_channels,
        eps=old_bn1.eps,
        momentum=old_bn1.momentum
    )

    # 3. 创建新的 Conv2
    block.conv2 = nn.Conv2d(
        in_channels=mid_channels,
        out_channels=old_conv2.out_channels,
        kernel_size=old_conv2.kernel_size,
        stride=old_conv2.stride,
        padding=old_conv2.padding,
        dilation=old_conv2.dilation,
        groups=old_conv2.groups,
        bias=old_conv2.bias is not None
    )


#加载函数

def load_pruned_resnet34(checkpoint_path):

    # 1. 读取保存的 Checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    # 2. 确认文件属于我们当前的模型格式
    if checkpoint["model_type"] != "cifar_resnet34":
        raise ValueError("Checkpoint 的模型类型不正确")

    # 3. 先创建原始 ResNet34 结构
    model = resnet.resnet34(
        num_classes=checkpoint["num_classes"]
    )

    # 4. 读取各个 BasicBlock 的通道配置
    channel_config = checkpoint["channel_config"]

    expected_keys = {
        f"{layer_name}.{i}"
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]
        for i in range(len(getattr(model, layer_name)))
    }

    if set(channel_config.keys()) != expected_keys:
        raise ValueError("通道配置与 ResNet34 不匹配")

    # 5. 逐个重建 BasicBlock
    for key, mid_channels in channel_config.items():

        layer_name, index = key.split(".")

        layer = getattr(model, layer_name)
        block = layer[int(index)]

        if (
            type(mid_channels) is not int
            or not 1 <= mid_channels <= block.conv1.out_channels
        ):
            raise ValueError(f"无效通道数: {key}")

        rebuild_block(block, mid_channels)

    # 6. 加载真正的剪枝权重
    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True
    )

    # 7. 设置推理模式
    model.eval()

    return model
