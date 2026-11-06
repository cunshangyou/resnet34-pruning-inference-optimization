
import os
import time
import multiprocessing as mp

import psutil

# ==========================================
# 1. 五组模型配置
# ==========================================

MODEL_CONFIGS = {
    "Original": "original",

    "Pruned_75":
        "outputs/inference_only/Pruned_75.pth",

    "Finetuned_75":
        "outputs/inference_only/Finetuned_75.pth",

    "Pruned_50":
        "outputs/inference_only/Pruned_50.pth",

    "Finetuned_50":
        "outputs/inference_only/Finetuned_50.pth"
}

def memory_worker(
    model_name,
    model_path,
    ready_queue,
    result_queue,
    start_event,
    done_event
):

    import gc
    import torch
    import resnet

    from load_pruned import load_pruned_resnet34

    torch.set_num_threads(4)

    process = psutil.Process(os.getpid())

    # 1. 记录 PyTorch 导入后的基础内存
    baseline_rss = (
        process.memory_info().rss / (1024 ** 2)
    )

    # ======================================
    # 2. 加载模型
    # ======================================

    if model_name == "Original":

        model = resnet.resnet34(num_classes=10)

        checkpoint = torch.load(
            "pretrained/cifar10_resnet34.pth",
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        # 释放 Checkpoint，避免重复占用内存
        del checkpoint

        gc.collect()

    else:

        model = load_pruned_resnet34(model_path)

        gc.collect()

    model.eval()

    # 3. 记录模型加载后的内存
    loaded_rss = (
        process.memory_info().rss / (1024 ** 2)
    )

    # ======================================
    # 4. 准备输入并预热
    # ======================================

    x = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():

        for _ in range(20):
            model(x)

    after_warmup_rss = (
        process.memory_info().rss / (1024 ** 2)
    )

    # 5. 通知主进程：模型已加载、预热完成
    ready_queue.put({
        "baseline_rss": baseline_rss,
        "loaded_rss": loaded_rss,
        "after_warmup_rss": after_warmup_rss
    })

    # 等待主进程完成初始内存采样
    start_event.wait()

    # ======================================
    # 6. 正式推理
    # ======================================

    with torch.inference_mode():

        for _ in range(200):
            model(x)

    # 推理完成后记录内存
    inference_end_rss = (
        process.memory_info().rss / (1024 ** 2)
    )

    done_event.set()

    # ======================================
    # 7. 返回结果
    # ======================================

    result_queue.put({
        "model": model_name,
        "baseline_rss": baseline_rss,
        "loaded_rss": loaded_rss,
        "after_warmup_rss": after_warmup_rss,
        "inference_end_rss": inference_end_rss
    })


# ==========================================
# 3. 主进程：监控子进程 RSS
# ==========================================


def test_memory(model_name, model_path):

    import queue

    ready_queue = mp.Queue()
    result_queue = mp.Queue()

    start_event = mp.Event()
    done_event = mp.Event()

    process = mp.Process(
        target=memory_worker,
        args=(
            model_name,
            model_path,
            ready_queue,
            result_queue,
            start_event,
            done_event
        )
    )

    process.start()

    child = psutil.Process(process.pid)

    # 整个进程运行期间的观测峰值
    overall_peak_rss = 0.0

    # ======================================
    # 1. 等待模型加载，同时开始监控
    # ======================================

    start_time = time.time()
    ready = None

    while ready is None:

        try:
            current_rss = (
                child.memory_info().rss / (1024 ** 2)
            )

            overall_peak_rss = max(
                overall_peak_rss,
                current_rss
            )

        except psutil.NoSuchProcess:
            raise RuntimeError(
                f"{model_name} 进程提前退出"
            )

        try:
            ready = ready_queue.get(timeout=0.005)

        except queue.Empty:

            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"{model_name} 加载失败"
                )

            if time.time() - start_time > 600:
                process.terminate()
                process.join()
                raise TimeoutError(
                    f"{model_name} 加载超时"
                )

    # ======================================
    # 2. 获取加载与预热数据
    # ======================================

    baseline_rss = ready["baseline_rss"]
    loaded_rss = ready["loaded_rss"]
    after_warmup_rss = ready["after_warmup_rss"]

    # 纳入已知内存采样值
    overall_peak_rss = max(
        overall_peak_rss,
        baseline_rss,
        loaded_rss,
        after_warmup_rss
    )

    # 推理阶段峰值至少包含预热后的 RSS
    inference_peak_rss = after_warmup_rss

    # ======================================
    # 3. 开始正式推理
    # ======================================

    # 告诉子进程：主进程已经开始监控
    start_event.set()

    while not done_event.is_set():

        try:

            current_rss = (
                child.memory_info().rss / (1024 ** 2)
            )

            overall_peak_rss = max(
                overall_peak_rss,
                current_rss
            )

            inference_peak_rss = max(
                inference_peak_rss,
                current_rss
            )

        except psutil.NoSuchProcess:
            break

        time.sleep(0.005)

    # ======================================
    # 4. 获取最终结果
    # ======================================

    result = result_queue.get(timeout=30)

    process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"{model_name} 测试进程异常退出"
        )

    # 合并所有已知峰值
    overall_peak_rss = max(
        overall_peak_rss,
        result["inference_end_rss"]
    )

    inference_peak_rss = max(
        inference_peak_rss,
        result["inference_end_rss"]
    )

    # ======================================
    # 5. 返回统计数据
    # ======================================

    return {
        "baseline": baseline_rss,
        "loaded": loaded_rss,
        "after_warmup": after_warmup_rss,

        "peak": overall_peak_rss,
        "inference_peak": inference_peak_rss,

        "load_increment": (
            loaded_rss - baseline_rss
        ),

        "inference_increment": max(
            0,
            inference_peak_rss - after_warmup_rss
        )
    }


# ==========================================
# 4. 测试五个模型
# ==========================================

if __name__ == "__main__":

    results = {}

    print("=" * 65)
    print("ResNet34 CPU 内存测试")
    print("=" * 65)

    for name, path in MODEL_CONFIGS.items():

        print(f"\n正在测试: {name}")

        result = test_memory(name, path)

        results[name] = result

        print(
            f"Baseline RSS: "
            f"{result['baseline']:.2f} MiB"
        )

        print(
            f"Loaded RSS: "
            f"{result['loaded']:.2f} MiB"
        )

        print(
            f"Peak RSS: "
            f"{result['peak']:.2f} MiB"
        )

        print(
            f"Model Load Increment: "
            f"{result['load_increment']:.2f} MiB"
        )

        print(
            f"Inference Peak Increment: "
            f"{result['inference_increment']:.2f} MiB"
        )
        print(
    f"After Warmup RSS: "
    f"{result['after_warmup']:.2f} MiB"
)

        print(
    f"Inference Peak RSS: "
    f"{result['inference_peak']:.2f} MiB"
)

    # ======================================
    # 5. 最终结果汇总
    # ======================================

    print("\n" + "=" * 85)
    print("最终内存测试结果")
    print("=" * 85)

    print(
        f"{'Model':<18}"
        f"{'Baseline':<14}"
        f"{'Loaded':<14}"
        f"{'Peak RSS':<14}"
        f"{'Load Inc.':<14}"
        f"{'Infer Inc.':<14}"
    )

    for name, result in results.items():

        print(
            f"{name:<18}"
            f"{result['baseline']:<14.2f}"
            f"{result['loaded']:<14.2f}"
            f"{result['peak']:<14.2f}"
            f"{result['load_increment']:<14.2f}"
            f"{result['inference_increment']:<14.2f}"
        )

    print("=" * 85)
    print("内存测试完成！")
