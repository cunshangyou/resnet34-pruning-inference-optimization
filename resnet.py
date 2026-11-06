# ResNet for CIFAR (32x32)
# 2019.07.24-Changed output of forward function
# Huawei Technologies Co., Ltd. <foss@huawei.com>
# taken from https://github.com/huawei-noah/Data-Efficient-Model-Compression/blob/master/DAFL/resnet.py
# for comparison with DAFL


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):#主管浅层残差网络，下面那个类管深层残差网络
    expansion = 1#表示最终输出通道数是planes的1倍
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        #stride = stride ：同一个代码模板，通过外部传入不同的参数（1 或 2），自动适配不同的特征图尺寸变化需求
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)#归一化，防止梯度消失
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:#表示两个通道数不同，需要处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        #总的来说就是为了避免stride=2时把通道扩展了从而不能做加法而扩展通道
    def forward(self, x):#这段 forward 代码是整个残差块（Residual Block）的执行说明书
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))#这一行代码与上一行代码是卷积提取特征
        out += self.shortcut(x)
        #为什么要shortcut（x）呢？是因为如果卷积提取特征很失败，加上原来的数据可以解决网络退化的问题
        #out 是主分支学出来的“残差特征”，self.shortcut(x) 是捷径分支传过来的“原始特征”
        out = F.relu(out)#激活函数，交给下一层
        return out
    
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)#（没有 ReLU！ 升维后保留负数）
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):#block指定使用什么残差模块 和 num_blocks最终分类类别数，具体看最下面的定义代码
        super(ResNet, self).__init__()
        self.in_planes = 64 #记录下一次创建 BasicBlock 时，输入特征图应该有多少个通道
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)#3
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)#4
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)#6
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)#3
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)#创建需求量个BasicBlock
        layers = []
        for stride in strides:#这个设计可以保证新创建的每个 Block 的输入通道数与前一个 Block 的输出通道数一致
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))#把每个通道的空间特征总结成一个代表值
        feature = out.view(out.size(0), -1)#这个操作不会改变元素的数值，只是重新组织 Tensor 的 Shape
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out
 
# =========================================================================
# 🌟 整体说明：
# ResNet 的主类 ResNet 接收三个核心参数：
#   1. 积木类型 (block)：用 BasicBlock（2层）还是 Bottleneck（3层）
#   2. 积木数量 (layers)：4个阶段(Stage 1~4)各自堆叠的积木个数 [s1, s2, s3, s4]
#   3. 分类类别数 (num_classes)：默认10类
# 
# 💡 总层数计算公式：1 + (积木总数 * 单块积木层数) + 1
#    (开头的1层7x7大卷积 + 中间的积木层 + 结尾的1层全连接层)
# =========================================================================

def resnet18(num_classes=10):
    """
    ResNet18：轻量级网络，速度快，适合简单任务或算力受限场景。
    - 积木类型：BasicBlock（单块积木包含 2 个 3x3 卷积层）
    - 积木数量：[2, 2, 2, 2] 表示 4 个阶段各堆叠 2 块积木，共 8 块。
    - 层数计算：1 + (8 * 2) + 1 = 18 层
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    """
    ResNet34：中等深度，精度与速度平衡，常用于常规图像分类。
    - 积木类型：BasicBlock（单块积木包含 2 个 3x3 卷积层）
    - 积木数量：[3, 4, 6, 3] 表示 4 个阶段各堆叠 3,4,6,3 块积木，共 16 块。
    - 层数计算：1 + (16 * 2) + 1 = 34 层
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


# ⚠️ 注意：下面开始换积木了！BasicBlock 变成了 Bottleneck
# Bottleneck 单块积木包含 3 个卷积层（1x1, 3x3, 1x1），且通道数会膨胀 4 倍 (expansion=4)

def resnet50(num_classes=10):
    """
    ResNet50：深度网络的起点，使用瓶颈结构极大节省了算力。
    - 积木类型：Bottleneck（单块积木包含 3 个卷积层）
    - 积木数量：[3, 4, 6, 3]（虽然数量跟 ResNet34 一样，但变成了 3 层结构）
    - 层数计算：1 + (16 * 3) + 1 = 50 层
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=10):
    """
    ResNet101：加深网络，在复杂任务上精度更高，但计算耗时也更长。
    - 积木类型：Bottleneck（单块积木包含 3 个卷积层）
    - 积木数量：[3, 4, 23, 3] 主要在第三个阶段(Stage 3)疯狂堆叠了 23 块。
    - 层数计算：1 + (33 * 3) + 1 = 101 层
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=10):
    """
    ResNet152：超深网络，计算量极大，通常用于算力充沛、追求极致精度的场景。
    - 积木类型：Bottleneck（单块积木包含 3 个卷积层）
    - 积木数量：[3, 8, 36, 3] 进一步在各个阶段狂堆积木，总共 50 块。
    - 层数计算：1 + (50 * 3) + 1 = 152 层
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

















import torch
import resnet

from thop import profile

model = resnet.resnet34(num_classes=10)
model.eval()

# 1. 统计参数量
params = sum(p.numel() for p in model.parameters())

print(f"Parameters: {params:,}")
print(f"Parameters: {params / 1e6:.4f} M")

# 2. 估算 FP32 参数存储成本
size_mb = params * 4 / 1e6

print(f"FP32 Parameter Size: {size_mb:.2f} MB")

# 3. 统计 MACs
x = torch.randn(1, 3, 32, 32)

with torch.inference_mode():
    macs, thop_params = profile(
        model,
        inputs=(x,),
        verbose=False
    )

print(f"MACs: {macs / 1e9:.4f} G")
print(f"THOP Parameters: {thop_params / 1e6:.4f} M")