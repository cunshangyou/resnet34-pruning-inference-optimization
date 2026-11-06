
import sys
import os
import platform
import subprocess
import importlib.metadata

import torch
import torchvision
import psutil


# ==========================================
# 1. 操作系统信息
# ==========================================

print("=" * 60)
print("实验环境信息")
print("=" * 60)

print("\n【操作系统】")

print("OS:", platform.system())
print("OS Version:", platform.version())
print("Platform:", platform.platform())


# ==========================================
# 2. Python 环境
# ==========================================

print("\n【Python 环境】")

print("Python Version:", sys.version)
print("Python Executable:", sys.executable)


# ==========================================
# 3. 深度学习框架
# ==========================================

print("\n【深度学习框架】")

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

for package in ["thop", "numpy", "psutil"]:

    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")

    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: Not Installed")


# ==========================================
# 4. CPU 硬件信息
# ==========================================

print("\n【CPU 信息】")

print("CPU:", platform.processor())

print("Logical CPU Cores:", psutil.cpu_count(logical=True))

print("Physical CPU Cores:", psutil.cpu_count(logical=False))


# ==========================================
# 5. PyTorch CPU 配置
# ==========================================

print("\n【PyTorch 配置】")

print("CPU Threads:", torch.get_num_threads())

print("Inter-op Threads:", torch.get_num_interop_threads())

print("MKLDNN Available:", torch.backends.mkldnn.is_available())

print("MKLDNN Enabled:", torch.backends.mkldnn.enabled)

print("CUDA Available:", torch.cuda.is_available())


# ==========================================
# 6. 实验 Benchmark 配置
# ==========================================

print("\n【实验配置】")

print("Device: CPU")
print("Batch Size: 1")
print("CPU Threads: 4")
print("Warmup: 20")
print("Repeat per Round: 100")
print("Benchmark Rounds: 5")

print("Input Shape: [1, 3, 32, 32]")
print("Data Type: FP32")


# ==========================================
# 7. Windows CPU 型号补充查询
# ==========================================

if platform.system() == "Windows":

    print("\n【Windows CPU 型号】")

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | "
                "Select-Object -First 1 -ExpandProperty Name)"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )

        print("CPU Model:", result.stdout.strip())

    except Exception as e:
        print("CPU Model Query Failed:", e)


# ==========================================
# 8. 内存信息
# ==========================================

memory = psutil.virtual_memory()

print("\n【系统内存】")

print(f"Total RAM: {memory.total / (1024**3):.2f} GB")

print(f"Available RAM: {memory.available / (1024**3):.2f} GB")


print("\n环境信息记录完成！")
