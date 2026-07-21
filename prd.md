# 产品需求文档 (PRD)：gpud 弹性显存调度框架

---

## 1. 产品概述

### 1.1 产品定位

`gpud` 是一个专为 PyTorch 分布式数据并行（DDP）设计的极轻量、声明式弹性 GPU 调度与显存换入换出（Offloading）框架。它允许用户在不中断训练进程的前提下，通过修改配置文件动态增减训练所使用的 GPU 数量。

### 1.2 核心解决痛点

* **动态增容减容**：支持在数据并行训练中动态指定任意 GPU 卡号与卡数量（例如 8 卡全量 `[0, 1, 2, 3, 4, 5, 6, 7]` 或缩容至 4 卡 `[0, 1, 2, 3]` 及任意卡数/卡号组合），实现训练过程中的平滑增容与减容。
* **重启成本高**：消除传统“手动杀进程 -> 修改可用卡数 -> 重新加载 Checkpoint”所带来的高昂 I/O 开销与时间浪费。
* **代码侵入性高**：避免用户在主训练逻辑中编写复杂的 DDP 进退群组、显存搬运与数据切分逻辑。

---

## 2. 系统架构与核心机制

### 2.1 架构设计

系统由三个核心部分完全解耦构成：

1. **控制层（`config.toml`）**：声明式状态机，用户通过修改文件下达扩缩容指令。
2. **调度层（`gpud.py`）**：以单文件装饰器（Decorator）形式存在，负责拦截 Epoch 边界、解析指令、重组 NCCL 通信群组、切分数据集及显存换页。
3. **业务层（`train.py`）**：用户原生的 PyTorch 训练脚本，保持纯粹的单步训练逻辑。

### 2.2 核心调度机制（Epoch-Bound Elastic Scaling）

* 调度操作严格限制在 **Epoch 边界**进行，绝对不打断单个 Batch 的算子执行及反向传播。
* 依赖 PyTorch `torch.distributed.new_group` 实现动态子网重组。

---

## 3. 详细功能需求 (Functional Requirements)

### FR-1：声明式资源配置

* **定义**：系统必须通过读取 `config.toml` 中的 `active_gpus` 数组来获取下一个 Epoch 期望使用的 GPU 列表。
* **规则**：
* 支持在 PyTorch 数据并行（DDP）训练中动态指定任意 GPU 卡号与任意卡数量，例如 8 卡全量 `[0, 1, 2, 3, 4, 5, 6, 7]`、缩容至 4 卡 `[0, 1, 2, 3]` 或任意卡数/卡号组合（如 `[0, 2, 5]` 等）。
* 主进程（Rank 0）负责在 Epoch 开始前实时读取此配置。



### FR-2：零侵入式装饰器接入 (`@elastic_scheduler`)

* **定义**：必须提供一个装饰器 `@elastic_scheduler(config_path)`，用户仅需将其挂载至按 Epoch 划分的训练函数上。
* **行为**：
* 装饰器接收原始模型（非 DDP 包装）与原始 Dataset。
* 装饰器在内部接管 DDP 的包装与 DataLoader 的生成。
* 运行时，装饰器将处理好的 `ddp_model`、`optimizer` 和 `dataloader` 注入回用户的训练函数。



### FR-3：动态通信群组重组 (Dynamic NCCL Grouping)

* **定义**：在每个 Epoch 开始前，系统根据 `active_gpus` 销毁旧群组并建立新的 NCCL 通信子组。
* **行为**：
* 只有在 `active_gpus` 列表中的 Rank 进程才会被纳入新的 `process_group` 进行后续的前向与反向传播。
* 退出子组的进程自动跳过当前 Epoch 的训练计算。



### FR-4：自适应数据平摊 (Auto Data Redistribution)

* **定义**：当参与训练的 GPU 数量发生变化时，必须保证整个数据集的正确平摊。
* **行为**：
* 装饰器需动态生成 `DistributedSampler`，其 `num_replicas` 等于当前 `active_gpus` 的长度。
* 确保缩容（如 8 变 4）时，当前 4 张卡平摊 100% 的数据量；扩容（4 变 8）时，平摊策略自动恢复，避免数据遗漏或重复。



### FR-5：物理显存挂起与恢复 (VRAM Offloading & Resume)

* **定义**：落选（不在 `active_gpus` 列表中）的进程必须彻底释放 GPU 物理显存。
* **行为**：
* **换出（Offload）**：针对落选进程，将其模型参数（Parameters/Buffers）及优化器状态（Optimizer States，如动量）全量转移至 CPU RAM。
* **清空（Empty）**：执行 Python `gc.collect()` 与 `torch.cuda.empty_cache()`，确保目标 GPU 显存占用绝对降至 **0 MB**。
* **恢复（Resume）**：当该进程在未来 Epoch 重新被加入 `active_gpus` 时，自动将上述数据从 CPU RAM 重载回目标 CUDA 设备。



---

## 4. 非功能需求 (Non-Functional Requirements)

### NFR-1：性能与时延

* 在 PCIe 5.0 环境下，单节点 8 张 H100 级别的模型（如 ViT-H，数百 MB 至数 GB 显存占用）换出至 CPU RAM 及清空操作，总耗时不得超过 **3 秒**。
* 重载恢复耗时不得超过 **3 秒**。

### NFR-2：稳定性与一致性

* 多次触发扩容与缩容（如 `8 -> 4 -> 8 -> 2 -> 8`），不可发生 CPU RAM 或 GPU VRAM 的内存泄漏（Memory Leak）。
* 全局 Epoch 计数器、学习率调度步数不受扩缩容操作影响，保证严密的训练收敛等价性。

### NFR-3：兼容性

* 仅依赖 PyTorch 原生 API (`torch.distributed`) 及 Python 标准库 (`os`, `ast`, `gc`, `functools`)，禁止引入任何 C++ 编译扩展或其他第三方重型依赖。

---

## 5. 用户操作交互流 (User Flow)

1. **环境准备**：用户编写标准单机版的 Dataset 与 Model 逻辑，不含任何 DDP 相关侵入代码。
2. **接入调度**：用户使用 `@elastic_scheduler` 包裹 `train_one_epoch` 函数，通过 `torchrun --nproc_per_node=8` 启动训练。
3. **动态缩容（让卡）**：
* 训练进行中，用户手动编辑 `config.toml`，将 `active_gpus` 改为 `[0, 1, 2, 3]`。
* 当前 Epoch 结束，卡 4~7 将自动挂起，显存降至 0 MB，释放给他人。
* 卡 0~3 自动调整 DataLoader 占比，无缝继续下一个 Epoch。


4. **动态扩容（收回卡）**：
* 其他人使用完毕，用户修改 `config.toml` 恢复为 `[0, 1, 2, 3, 4, 5, 6, 7]`。
* 当前 Epoch 结束，卡 4~7 将模型重新载入显存，8 卡重组集群，加速继续训练。