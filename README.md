
# ResNet34 Pruning and Inference Optimization

**资源受限设备上的深度神经网络推理优化设计与实现**

基于 PyTorch 和 CIFAR-10 的 ResNet34 结构化剪枝、微调与 CPU 推理性能优化实验。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14.4-blue" />
  <img src="https://img.shields.io/badge/PyTorch-2.15.0-orange" />
  <img src="https://img.shields.io/badge/Dataset-CIFAR--10-green" />
  <img src="https://img.shields.io/badge/Method-Structured%20Pruning-purple" />
</p>

## 1. 项目简介 | Introduction

本项目面向资源受限设备上的深度神经网络推理优化，以 CIFAR-10 图像分类任务和 ResNet34 为研究对象，基于教师提供的网络实现、预训练权重及辅助工具，完成模型基线评估、结构化通道剪枝、剪枝后微调和 CPU 推理性能测试。

项目主要研究目标：

- **Model Compression：** 通过结构化剪枝减少模型参数量、理论计算量和权重存储空间。
- **Accuracy Recovery：** 通过 Fine-tuning 恢复剪枝造成的分类准确率损失。
- **Inference Optimization：** 评估结构化剪枝对 CPU 实际推理延迟的改善效果。
- **Comprehensive Evaluation：** 从准确率、参数量、MACs、模型权重大小、推理延迟和进程内存等多个维度评价优化效果。

本项目设置两组独立剪枝实验，分别保留 BasicBlock 内部 75% 和 50% 的通道，并对剪枝后的模型进行微调。

### 核心实验成果

| 指标            |   原始模型 | 75% 通道保留（微调后） | 50% 通道保留（微调后） |
| --------------- | ---------: | ---------------------: | ---------------------: |
| Top-1 Accuracy  |     95.70% |                 95.33% |                 94.86% |
| Parameters      |  21.2821 M |              16.0087 M |              10.7352 M |
| MACs            |   1.1635 G |               0.8752 G |               0.5869 G |
| State Dict Size |   85.26 MB |               64.16 MB |               43.06 MB |
| CPU Latency     | 17.8246 ms |             14.1417 ms |             10.4220 ms |
| Speedup         |    1.000× |                1.260× |                1.710× |

**主要结论：**

- 75% 通道保留方案：参数量减少 24.78%，CPU 推理延迟降低 20.66%，微调后准确率为 95.33%。
- 50% 通道保留方案：参数量减少 49.56%，CPU 推理延迟降低 41.53%，微调后准确率为 94.86%。

在本实验的 CPU 测试环境下，50% 通道保留方案取得了更明显的推理加速，同时通过微调恢复了大部分分类精度。

## 2. 技术路线 | Methodology

整体实验流程：

```text
        CIFAR-10 Dataset
                |
                v
       Pretrained ResNet34
                |
                v
       Baseline Evaluation
                |
                v
      Structured Channel Pruning
                |
          +-----+-----+
          |           |
          v           v
    KEEP_RATIO=0.75  KEEP_RATIO=0.50
          |           |
          v           v
     Pruned Model  Pruned Model
          |           |
          v           v
      Fine-tuning  Fine-tuning
          |           |
          +-----+-----+
                |
                v
       Model Performance Test
                |
                v
 Accuracy / Parameters / MACs
 State Dict / Latency / RSS
```

### 2.1 Baseline Model

- Network: ResNet34
- Dataset: CIFAR-10
- Training samples: 50,000
- Test samples: 10,000
- Original Accuracy: 95.70%
- Original Parameters: 21.2821 M
- Original MACs: 1.1635 G

### 2.2 Structured Pruning

采用基于 Filter Weight L1-Norm 的结构化通道剪枝方法。

核心思想：

1. 计算 BasicBlock 中 Conv1 每个输出 Filter 的 L1-Norm。
2. 根据重要性分数选择需要保留的通道。
3. 同步保留 BN1 对应的 affine parameters 和 running statistics。
4. 在 Conv2 的输入通道维度选择相同索引。
5. 保持 BasicBlock 外部输入、输出通道数以及 BN2 和 Shortcut 不变。

通过上述操作，在减少 BasicBlock 内部通道的同时，保持残差连接的张量形状一致。

核心代码：

```python
# Calculate the L1-Norm of each output filter
scores = old_conv1.weight.detach().abs().sum(
    dim=(1, 2, 3)
)

# Select important channels
keep_idx = torch.argsort(
    scores, descending=True
)[:keep_channels]

keep_idx = torch.sort(keep_idx).values

# Prune Conv1 output channels
new_conv1.weight.copy_(
    old_conv1.weight[keep_idx]
)

# Synchronize Conv2 input channels
new_conv2.weight.copy_(
    old_conv2.weight[:, keep_idx, :, :]
)

# Synchronize BatchNorm parameters
new_bn1.weight.copy_(
    old_bn1.weight[keep_idx]
)

new_bn1.bias.copy_(
    old_bn1.bias[keep_idx]
)

new_bn1.running_mean.copy_(
    old_bn1.running_mean[keep_idx]
)

new_bn1.running_var.copy_(
    old_bn1.running_var[keep_idx]
)
```

### 2.3 Pruning Configurations

本实验采用两种通道保留比例：

| Configuration | Keep Ratio | Removed Channels |
| ------------- | ---------: | ---------------: |
| Pruning 75%   |        75% |              25% |
| Pruning 50%   |        50% |              50% |

两组实验均从相同的原始预训练模型独立剪枝。

### 2.4 Fine-tuning

剪枝会改变模型结构，可能造成分类准确率下降，因此需要对剪枝模型进行微调。

训练配置：

| Configuration | 75% Keep         | 50% Keep         |
| ------------- | ---------------- | ---------------- |
| Optimizer     | SGD              | SGD              |
| Momentum      | 0.9              | 0.9              |
| Weight Decay  | 1e-4             | 1e-4             |
| Batch Size    | 64               | 64               |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |
| Learning Rate | 0.001            | 0.01 → 0.001    |
| Epochs        | 3                | 3 + 5 = 8        |

训练集数据增强：

- RandomCrop (padding=4)
- RandomHorizontalFlip
- ToTensor
- CIFAR-10 Standardization

测试集仅进行 ToTensor 和标准化。

### 2.5 Model Checkpoint and Recovery

剪枝模型保存以下信息：

- model_type
- num_classes
- keep_ratio
- channel_config
- state_dict

模型加载时，根据 channel_config 重建 BasicBlock 的内部通道结构，再通过 `strict=True` 加载权重。

通过参数量检查、输出 Shape 检查以及保存前后输出一致性测试验证模型恢复过程。

## 3. 实验结果 | Experimental Results

所有结果均来自本项目实验报告中的实际测试数据。

### 3.1 Classification Accuracy

| Model               | Top-1 Accuracy |
| ------------------- | -------------: |
| Original ResNet34   |         95.70% |
| Pruned 75% Keep     |         95.12% |
| Fine-tuned 75% Keep |         95.33% |
| Pruned 50% Keep     |         84.06% |
| Fine-tuned 50% Keep |         94.86% |

实验分析：

- 75% 通道保留模型剪枝后准确率为 95.12%，微调后达到 95.33%，与原始模型仅相差 0.37 个百分点。
- 50% 通道保留模型剪枝后准确率下降至 84.06%，经过累计 8 Epoch 微调后恢复至 94.86%。
- 50% 通道保留模型恢复了约 92.78% 的剪枝准确率损失。

### 3.2 Parameters and MACs

| Model          | Parameters (M) | MACs (G) |
| -------------- | -------------: | -------: |
| Original       |        21.2821 |   1.1635 |
| Pruned 75%     |        16.0087 |   0.8752 |
| Fine-tuned 75% |        16.0087 |   0.8752 |
| Pruned 50%     |        10.7352 |   0.5869 |
| Fine-tuned 50% |        10.7352 |   0.5869 |

相对于原始模型：

| Metric              | 75% Keep | 50% Keep |
| ------------------- | -------: | -------: |
| Parameter Reduction |   24.78% |   49.56% |
| MACs Reduction      |   24.78% |   49.55% |

Fine-tuning 不改变网络结构，因此微调前后参数量和 MACs 保持一致。

### 3.3 Serialized Model Weight Size

为了避免将训练 Checkpoint 中的优化器状态混入模型权重比较，本实验统一提取 State Dict，并采用相同的 `torch.save` 序列化方式测量。

| Model          | State Dict Size |
| -------------- | --------------: |
| Original       |        85.26 MB |
| Pruned 75%     |        64.16 MB |
| Fine-tuned 75% |        64.16 MB |
| Pruned 50%     |        43.06 MB |
| Fine-tuned 50% |        43.06 MB |

权重文件大小减少：

- 75% Keep：约 24.75%
- 50% Keep：约 49.50%

**注意：** 完整训练 Checkpoint 还包含 optimizer_state_dict 等训练状态，因此文件会明显大于纯 State Dict。

例如：

- Fine-tuned 75% 完整 Checkpoint：128.24 MB
- Fine-tuned 50% 完整 Checkpoint：86.05 MB

这并不意味着推理模型权重本身变大。

### 3.4 CPU Inference Latency

测试采用五轮 Benchmark 汇总结果。

| Model          | Median Latency (ms) | Speedup |
| -------------- | ------------------: | ------: |
| Original       |             17.8246 | 1.000× |
| Pruned 75%     |             14.3856 | 1.239× |
| Fine-tuned 75% |             14.1417 | 1.260× |
| Pruned 50%     |             10.5406 | 1.691× |
| Fine-tuned 50% |             10.4220 | 1.710× |

实际推理性能提升：

| Model          | Latency Reduction | Speedup |
| -------------- | ----------------: | ------: |
| Fine-tuned 75% |            20.66% | 1.260× |
| Fine-tuned 50% |            41.53% | 1.710× |

50% 通道保留方案在本次 CPU 测试中获得了最明显的推理加速。

实际延迟改善幅度与 MACs 减少幅度并不完全相同，说明算子实现、内存访问和线程调度同样会影响实际推理性能。

### 3.5 CPU Memory Usage

每个模型在独立子进程中测量 RSS。

| Model          | Loaded RSS (MiB) | Peak RSS (MiB) |
| -------------- | ---------------: | -------------: |
| Original       |           470.35 |         472.51 |
| Pruned 75%     |           443.57 |         459.85 |
| Fine-tuned 75% |           447.21 |         462.52 |
| Pruned 50%     |           437.28 |         443.52 |
| Fine-tuned 50% |           439.03 |         445.27 |

说明：

- RSS 是进程级内存指标，包含 Python、PyTorch 运行时、模型参数及其他内存。
- RSS 不等同于纯激活值内存。
- Peak RSS 通过 5 ms 采样得到，是观测峰值，并非精确的激活值峰值内存。

### 3.6 Overall Comparison

综合实验结果：

| Metric           | Original | Fine-tuned 75% | Fine-tuned 50% |
| ---------------- | -------: | -------------: | -------------: |
| Accuracy         |   95.70% |         95.33% |         94.86% |
| Parameters (M)   |  21.2821 |        16.0087 |        10.7352 |
| MACs (G)         |   1.1635 |         0.8752 |         0.5869 |
| State Dict (MB)  |    85.26 |          64.16 |          43.06 |
| Latency (ms)     |  17.8246 |        14.1417 |        10.4220 |
| Speedup          |  1.000× |        1.260× |        1.710× |
| Loaded RSS (MiB) |   470.35 |         447.21 |         439.03 |

**Experimental Conclusion:**

结构化通道剪枝能够有效降低 ResNet34 的参数量、理论计算量和模型权重存储空间，并在本实验的 CPU 环境下带来实际推理加速。

50% 通道保留方案在模型压缩和推理速度方面效果更加明显，但需要更多微调训练来恢复准确率。

## 4. 项目结构 | Project Structure

```text
pruning_demo/
│
├── datasets/
│   └── cifar10/
│       └── cifar-10-batches-py/
│
├── pretrained/
│   └── cifar10_resnet34.pth
│
├── outputs/
│   ├── inference_only/
│   ├── resnet34_pruned_50_finetuned.pth
│   ├── resnet34_pruned_50_finetuned_stage2.pth
│   ├── resnet34_pruned_50_v2.pth
│   ├── resnet34_pruned_75.pth
│   ├── resnet34_pruned_75_finetuned.pth
│   └── resnet34_pruned_75_v2.pth
│
├── resnet.py
├── utils.py
├── prune_model.py
├── load_pruned.py
│
├── finetune_pruned.py
├── finetune_pruned_75.py
├── finetune_continue.py
├── prepare_inference_checkpoints.py
│
├── environment_info.py
│
├── test_accuracy.py
├── test_pruned_accuracy.py
├── test_finetuned.py
├── test_finetuned_stage2.py
├── test_load_pruned.py
├── test_load_75_finetuned.py
├── test_resnet.py
│
├── test_model_size.py
├── test_memory.py
├── test_performance.py
├── test_finetuned_performance.py
├── test_all_models_performance.py
├── test_benchmark_multiround.py
│
├── pruning_pytorch_tools.ipynb
├── pruning_pytorch_tools.pdf
│
├── 资源受限设备上的深度神经网络推理优化实验报告.docx
├── .gitignore
└── .gitattributes
```

### Main Scripts

| File                                 | Function                               |
| ------------------------------------ | -------------------------------------- |
| `resnet.py`                        | CIFAR-10 ResNet34 网络定义             |
| `utils.py`                         | 数据加载、预处理和辅助评估             |
| `prune_model.py`                   | L1-Norm 结构化通道剪枝                 |
| `load_pruned.py`                   | 根据 channel_config 重建并加载剪枝模型 |
| `finetune_pruned.py`               | 50% 通道保留模型微调                   |
| `finetune_pruned_75.py`            | 75% 通道保留模型微调                   |
| `finetune_continue.py`             | 继续微调                               |
| `prepare_inference_checkpoints.py` | 准备推理 Checkpoint                    |
| `test_accuracy.py`                 | 原始模型准确率测试                     |
| `test_pruned_accuracy.py`          | 剪枝模型准确率测试                     |
| `test_all_models_performance.py`   | 五组模型性能对比                       |
| `test_model_size.py`               | 模型权重大小测试                       |
| `test_memory.py`                   | RSS 内存测试                           |
| `test_benchmark_multiround.py`     | 多轮 CPU Benchmark                     |

## 5. 实验环境 | Environment

### Hardware

| Component      | Configuration                           |
| -------------- | --------------------------------------- |
| CPU            | AMD Ryzen 9 8940HX with Radeon Graphics |
| Physical Cores | 16                                      |
| Logical Cores  | 32                                      |
| System Memory  | 16 GB (15.22 GB recognized)             |
| GPU / CUDA     | Not available                           |

### Software

| Component        | Version                  |
| ---------------- | ------------------------ |
| Operating System | Windows 11 (Build 26200) |
| Python           | 3.14.4                   |
| PyTorch          | 2.15.0+cpu               |
| Torchvision      | 0.29.2+cpu               |
| THOP             | 0.1.1-2209072238         |
| NumPy            | 2.4.6                    |
| psutil           | 7.2.2                    |

### Benchmark Configuration

| Parameter   | Configuration            |
| ----------- | ------------------------ |
| Device      | CPU                      |
| Threads     | 4                        |
| Precision   | FP32                     |
| Batch Size  | 1                        |
| Input Shape | [1, 3, 32, 32]           |
| Warmup      | 20 iterations            |
| Repeat      | 100 iterations per round |
| Rounds      | 5                        |

每轮预热 20 次，正式计时 100 次，计算每轮中位数，再取五轮中位数的中位数作为最终汇总结果。

## 6. Getting Started

### 6.1 Clone Repository

本项目使用 Git LFS 管理大型模型权重和 CIFAR-10 数据文件。

首先安装 Git LFS，然后执行：

```bash
git lfs install

git clone https://github.com/cunshangyou/resnet34-pruning-inference-optimization.git

cd resnet34-pruning-inference-optimization

git lfs pull
```

### 6.2 Environment Setup

本实验环境及依赖版本见上方 Environment 部分。

运行前请确保安装对应版本的 PyTorch、Torchvision、THOP、NumPy 和 psutil。

### 6.3 Run Experiments

主要实验入口：

```text
environment_info.py
    环境检测

test_accuracy.py
    原始模型基线测试

prune_model.py
    结构化剪枝

finetune_pruned.py
finetune_pruned_75.py
    剪枝模型微调

test_all_models_performance.py
    多模型性能对比

test_memory.py
    RSS 内存测试

test_benchmark_multiround.py
    多轮 CPU 推理 Benchmark
```

具体执行参数以各 Python 脚本中的配置为准。

## 7. Limitations and Future Work

### Current Limitations

1. **Fine-tuning Configurations:** 75% 通道保留模型微调 3 Epoch，50% 通道保留模型微调 8 Epoch，两组训练配置不同，不能视为完全同条件的微调对照。
2. **Hardware Dependency:** CPU 推理延迟依赖 Ryzen 9 8940HX、4 线程、FP32 和 Batch Size=1 的测试条件，不能直接推广到其他设备。
3. **Benchmark Reproducibility:** 五轮 Benchmark 在同一进程内轮换测试，并非五次独立进程重启。
4. **Deployment:** 尚未完成 INT8/FP16 量化、ONNX Runtime 和真实嵌入式设备部署。

### Future Work

- 探索不同结构化剪枝策略及通道重要性评估方法。
- 进一步优化剪枝后微调策略。
- 研究结构化剪枝与 INT8/FP16 量化的结合。
- 探索 ONNX Runtime 等推理引擎。
- 在真实资源受限设备上验证模型的推理性能。
- 增加不同硬件平台和独立进程下的 Benchmark 实验。

## 8. Experiment Report

完整实验报告：

`资源受限设备上的深度神经网络推理优化实验报告.docx`

报告包含实验目的、方法设计、核心代码、实验结果图表、性能数据及实验分析。

## 9. License


```
This project is distributed under the
Educational and Non-Commercial Research License (ENCRL) v1.0.

See [LICENSE.md](LICENSE.md) for the complete terms.

Commercial use and unauthorized redistribution are prohibited
without prior written permission.

Third-party materials remain subject to their respective licenses.
```

---

**Project:** ResNet34 Pruning and Inference Optimization

**Framework:** PyTorch

**Dataset:** CIFAR-10

**Pruning Method:** Structured Channel Pruning (L1-Norm)

**Repository:** https://github.com/cunshangyou/resnet34-pruning-inference-optimization
