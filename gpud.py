"""
==============================================================================
gpud.py - 极轻量、声明式弹性 GPU 调度与显存 Offloading 框架
==============================================================================
功能描述：
1. 声明式调度：在 Epoch 边界读取 config.toml 中的 active_gpus 列表。
2. 动态 NCCL 通信组：利用 torch.distributed.new_group 重组参与计算的 GPU 子集群。
3. 物理显存彻底释放（Offload）：非活跃卡将模型权重与优化器状态移至 CPU RAM，并清空 CUDA Cache (0 MB 占用)。
4. 自适应数据平摊：使用 DistributedSampler 依据 active_gpus 长度自动调整数据切分规则。
5. 多线程数据加载：默认将 DataLoader 的 num_workers 设置为 CPU 核心数的一半 (half CPU cores)。
==============================================================================
"""

import os
import gc
import sys
import time
import functools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def load_config(config_path: str) -> dict:
    """
    读取 config.toml 配置文件（优先使用 Python 3.11+ 标准库 tomllib，兜底手写解析器）
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[gpud] 配置文件不存在: {config_path}")

    # 优先尝试 Python 3.11+ 原生 tomllib
    try:
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except (ImportError, Exception):
        pass

    # 兜底轻量解析逻辑 (无需任何第三方依赖)
    config = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    # 解析如 active_gpus = [0, 1, 2, 3]
                    content = val[1:-1].strip()
                    if content:
                        config[key] = [int(x.strip()) for x in content.split(",") if x.strip()]
                    else:
                        config[key] = []
    return config


class OffloadManager:
    """
    显存换入换出管理器：负责模型参数与优化器状态在 CPU RAM 和 GPU VRAM 之间的迁移与释放
    """

    @staticmethod
    def offload_to_cpu(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """将模型参数与优化器状态全量迁移至 CPU RAM，并完全清空 GPU 显存"""
        # 1. 解包 DDP 模型获取原生 Module
        unwrapped_model = model.module if isinstance(model, DDP) else model

        # 2. 将模型参数与 Buffer 移至 CPU
        unwrapped_model.to("cpu")

        # 3. 将优化器动量/状态 Tensor 移至 CPU
        if optimizer is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")

        # 4. 强制垃圾回收与 GPU 显存清空
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def resume_to_gpu(model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        """将模型参数与优化器状态重新重载回指定的 CUDA GPU 设备"""
        unwrapped_model = model.module if isinstance(model, DDP) else model

        # 1. 模型移回 GPU
        unwrapped_model.to(device)

        # 2. 优化器状态移回 GPU
        if optimizer is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)


class SubGroupManager:
    """
    NCCL 通信子组管理器：负责根据 active_gpus 动态创建与缓存 process_group
    """
    _cache = {}

    @classmethod
    def get_or_create_subgroup(cls, active_gpus: list) -> dist.ProcessGroup:
        sorted_ranks = sorted(active_gpus)
        key = tuple(sorted_ranks)

        if key not in cls._cache:
            # 注意：dist.new_group 必须由 WORLD 中的所有进程共同调用
            new_pg = dist.new_group(ranks=sorted_ranks)
            cls._cache[key] = new_pg

        return cls._cache[key]


def elastic_scheduler(config_path: str = "config.toml"):
    """
    【核心装饰器】零侵入挂载至 Epoch 训练函数上
    
    使用方法：
        @elastic_scheduler(config_path="config.toml")
        def train_epoch(model, dataset_or_dataloader, optimizer, criterion, epoch, ...):
            ...
    """
    def decorator(func):
        # 记录每个进程的 offload 状态与 DDP 包装实例
        state_store = {
            "is_offloaded": False,
            "ddp_model": None,
            "last_active_gpus": None,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 自动初始化 DDP (若尚未初始化)
            if not dist.is_initialized():
                if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    torch.cuda.set_device(local_rank)
                    dist.init_process_group(backend="nccl")
                else:
                    # 单机非 torchrun 模式回退执行
                    return func(*args, **kwargs)

            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", global_rank % torch.cuda.device_count()))
            current_device = torch.device(f"cuda:{local_rank}")

            # 2. Rank 0 读取 config.toml 并在全局 WORLD 组中广播 active_gpus
            active_gpus = []
            if global_rank == 0:
                try:
                    cfg = load_config(config_path)
                    active_gpus = cfg.get("active_gpus", list(range(world_size)))
                except Exception as e:
                    print(f"[gpud Warning] 读取配置文件失败 ({e})，默认保持全量 GPU 运行。")
                    active_gpus = list(range(world_size))

            # 广播 active_gpus 配置到所有节点进程
            broadcast_payload = [active_gpus]
            dist.broadcast_object_list(broadcast_payload, src=0)
            active_gpus = sorted(broadcast_payload[0])

            # 校验 active_gpus 合法性
            if not active_gpus:
                active_gpus = list(range(world_size))

            # 3. 动态建立/获取通信子组
            sub_group = SubGroupManager.get_or_create_subgroup(active_gpus)

            # 4. 解析参数中的 model, dataset/dataloader, optimizer
            model = None
            dataset = None
            dataloader_obj = None
            optimizer = None
            args_list = list(args)

            # 位置参数匹配
            for arg in args_list:
                if isinstance(arg, torch.nn.Module):
                    model = arg
                elif isinstance(arg, DataLoader):
                    dataloader_obj = arg
                    dataset = arg.dataset
                elif isinstance(arg, Dataset):
                    dataset = arg
                elif isinstance(arg, torch.optim.Optimizer):
                    optimizer = arg

            # 关键字参数匹配
            if model is None:
                model = kwargs.get("model")
            if dataset is None and dataloader_obj is None:
                dataloader_obj = kwargs.get("dataloader")
                if dataloader_obj is not None:
                    dataset = dataloader_obj.dataset
                else:
                    dataset = kwargs.get("dataset")
            if optimizer is None:
                optimizer = kwargs.get("optimizer")

            # 5. 判断当前 Rank 是否在活跃列表中
            is_active = global_rank in active_gpus

            if not is_active:
                # ------------------------------------------------------------------
                # 【缩容/让卡逻辑】：当前 Rank 不在 active_gpus 中 -> 彻底 Offload
                # ------------------------------------------------------------------
                if not state_store["is_offloaded"]:
                    if model is not None:
                        OffloadManager.offload_to_cpu(model, optimizer)
                    state_store["is_offloaded"] = True
                    state_store["ddp_model"] = None
                    print(f"[gpud Status] Rank {global_rank} (GPU {local_rank}) -> 缩容挂起中 | 物理显存已释放 (VRAM: 0 MB)")

                # 落选进程跳过当前 Epoch 的计算逻辑
                return None

            # ----------------------------------------------------------------------
            # 【训练/恢复逻辑】：当前 Rank 在 active_gpus 中 -> 重载显存并参与计算
            # ----------------------------------------------------------------------
            sub_rank = active_gpus.index(global_rank)
            sub_world_size = len(active_gpus)

            # 若先前被挂起过，先将模型和优化器从 CPU 重载会 CUDA 设备
            if state_store["is_offloaded"]:
                if model is not None:
                    OffloadManager.resume_to_gpu(model, optimizer, current_device)
                state_store["is_offloaded"] = False
                print(f"[gpud Status] Rank {global_rank} (GPU {local_rank}) -> 重新激活 | 模型权重与优化器已重载回 VRAM")

            # 获取原生 Module
            unwrapped_model = model.module if isinstance(model, DDP) else model

            # 重新封装 DDP（针对当前 active_gpus 的子组）
            if state_store["ddp_model"] is None or state_store["last_active_gpus"] != active_gpus:
                ddp_model = DDP(
                    unwrapped_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    process_group=sub_group,
                    find_unused_parameters=False
                )
                state_store["ddp_model"] = ddp_model
                state_store["last_active_gpus"] = active_gpus
            else:
                ddp_model = state_store["ddp_model"]

            # 提取 Epoch 序号（如果传入了的话）
            epoch_val = kwargs.get("epoch", 0)
            for arg in args_list:
                if isinstance(arg, int) and not isinstance(arg, bool):
                    epoch_val = arg
                    break

            # 动态调整 DataLoader 与 DistributedSampler
            elastic_dataloader = None
            if dataset is not None:
                # 使用一半 CPU 核心数进行数据并行加载
                cpu_workers = max(1, (os.cpu_count() or 2) // 2)

                sampler = DistributedSampler(
                    dataset,
                    num_replicas=sub_world_size,
                    rank=sub_rank,
                    shuffle=True,
                    seed=42 + epoch_val
                )
                sampler.set_epoch(epoch_val)

                batch_size = 64
                if dataloader_obj is not None and hasattr(dataloader_obj, "batch_size"):
                    batch_size = dataloader_obj.batch_size

                elastic_dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=cpu_workers,
                    pin_memory=True,
                    drop_last=False
                )

            # 替换原始参数中的 model 和 dataloader 后执行原函数
            new_args = []
            for arg in args_list:
                if isinstance(arg, torch.nn.Module):
                    new_args.append(ddp_model)
                elif isinstance(arg, (DataLoader, Dataset)):
                    new_args.append(elastic_dataloader if elastic_dataloader is not None else arg)
                else:
                    new_args.append(arg)

            if "model" in kwargs:
                kwargs["model"] = ddp_model
            if "dataloader" in kwargs:
                kwargs["dataloader"] = elastic_dataloader
            elif "dataset" in kwargs:
                kwargs["dataset"] = elastic_dataloader

            # 执行用户的原始训练逻辑
            return func(*new_args, **kwargs)

        return wrapper

    return decorator
