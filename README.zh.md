# gpud: 极轻量 PyTorch 弹性 GPU 动态扩缩容与显存 Offload 框架

`gpud` 是一个专为 PyTorch 分布式数据并行（DDP）设计的极轻量、声明式弹性 GPU 调度与显存换入换出（Offloading）框架。

在模型训练过程中，你**无需终止进程**、**无需重新加载 Checkpoint**，只需直接修改 `config.toml` 配置文件，即可在 Epoch 边界动态平滑地改变参与训练的 GPU 卡数与指定卡号（如从 8 卡全量缩容至 4 卡或任意指定卡号组合），同时将落选 GPU 的显存完全释放至 **0 MB**。

---

## 🌟 核心特性与解决痛点

1. **动态增容减容 (Dynamic Scaling)**：
   * 支持数据并行（DDP）训练中动态指定任意卡号与卡数量。例如 8 卡全量 `[0, 1, 2, 3, 4, 5, 6, 7]`、动态缩容至 4 卡 `[0, 1, 2, 3]` 或任意组合如 `[0, 2, 5]`。
2. **零重启与免 I/O 成本 (Zero Downtime)**：
   * 彻底消除传统“手动杀死进程 -> 修改可用卡数 -> 重新加载 Checkpoint”带来的开销与时间浪费。
3. **物理显存彻底释放至 0 MB (Complete VRAM Offload)**：
   * 缩容挂起的 GPU 进程会自动将其模型权重、Buffer 及优化器状态（如 AdamW 动量）转移至 CPU RAM，并自动执行 `gc.collect()` 与 `torch.cuda.empty_cache()`，将显存完全释放为 **0 MB** 供他人使用。
4. **零侵入式挂载 (Zero-Invasive Design)**：
   * 业务代码无需编写复杂的 DDP 进退通信组、显存搬运与数据切分逻辑。仅需在单 Epoch 训练函数上挂载 **1 行装饰器** `@elastic_scheduler("config.toml")`。
5. **自适应数据切分 (Auto Data Redistribution)**：
   * 动态生成 `DistributedSampler`，根据当前 active GPUs 数量自动平摊 100% 的数据集，保证训练数据的严密性与收敛一致性。
6. **优化数据加载 (Half CPU Workers)**：
   * 自动利用系统 **一半 CPU 核心数** (`os.cpu_count() // 2`) 进行多线程 DataLoader 并行数据加载，提升整体训练吞吐。

---

## 🏗️ 系统架构

系统由三个核心部分完全解耦构成：

```
┌─────────────────────────────────────────────────────────────┐
│                 控制层 (config.toml)                        │
│          声明式定义 active_gpus = [0, 1, 2, 3]              │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Epoch 边界实时读取与广播)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 调度层 (gpud.py)                            │
│        @elastic_scheduler 装饰器 (重组 NCCL / VRAM Offload)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ (注入 ddp_model & dataloader)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 业务层 (demo.py / train.py)                 │
│              原生 PyTorch 单 Epoch 训练函数                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 安装与准备
项目无需额外安装第三方重型依赖，仅依赖 PyTorch (`torch`, `torchvision`) 及 Python 标准库。

```bash
git clone https://github.com/your-org/gpud.git
cd gpud
```

### 2. 配置文件 `config.toml`
在项目根目录创建或编辑 `config.toml`：

```toml
# 定义当前 Epoch 参与训练的物理 GPU 卡号列表
active_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
```

### 3. 在已有代码中挂载装饰器 (仅需 1 行)

```python
from gpud import elastic_scheduler

# ==============================================================================
# 【接入 gpud】：仅需添加 1 行装饰器即可！
# ==============================================================================
@elastic_scheduler(config_path="config.toml")
def train_one_epoch(model, dataset, optimizer, criterion, epoch, device):
    model.train()
    for images, targets in dataset: # 这里的 dataset 已被自动注入为 DistributedDataLoader
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
```

### 4. 启动多卡训练
通过 `torchrun` 启动分布式训练（例如 8 卡服务器）：

```bash
torchrun --nproc_per_node=8 demo.py
```

### 5. 训练中动态体验扩缩容 (让卡 / 收回卡)
在训练运行过程中，你可以随时在另一个终端打开并修改 `config.toml`：

* **动态缩容（让卡给他人）**：
  修改为 `active_gpus = [0, 1, 2, 3]` 并保存。
  *当前 Epoch 结束后，卡 4~7 的进程将自动将模型和优化器挂起至 CPU RAM，GPU 显存瞬间降至 0 MB；卡 0~3 自动按 4 卡切分数据，继续训练。*

* **动态扩容（收回卡继续加速）**：
  修改为 `active_gpus = [0, 1, 2, 3, 4, 5, 6, 7]` 并保存。
  *当前 Epoch 结束后，卡 4~7 自动从 CPU 重载数据回 GPU，重组 8 卡 NCCL 通信组，无缝加速训练。*

---

## 💻 Demo 脚本说明 (`demo.py`)

`demo.py` 提供了一个完整、开箱即用的 Vision Transformer 大模型分布式训练示例，展示了 `gpud` 的完整特性：

* **模型架构 (ViT-H)**： Vision Transformer Heavy (`img_size=224`, `patch_size=16`, `in_chans=3`, `embed_dim=1280`, `depth=32`, ~6.3 亿参数)。显存占用大，可清晰观测缩容时落选 GPU 7.5GB+ 显存瞬间释放至 **0 MB** 的过程。
* **数据集处理**： CIFAR-10 全量数据集（50,000 张训练集，10,000 张测试集），通过双三次插值 (`Bicubic`) 上采样至 `224x224` 标准规格。
* **无限 Epoch 训练**： 无最大 Epoch 限制，采用 `while True:` 无限循环，方便开发者随时修改 `config.toml` 测试多次动态缩容与扩容。
* **`tqdm` 经典进度条**： 固定宽度 `ncols=150`，仅在 Rank 0 打印，实时显示 `train_loss`、`train_acc`、`val_loss`、`val_acc`、运行耗时、it/s 速率及 ETA 剩余时间。
* **测试集评估 (Validation Acc)**： 每轮训练后自动对 10,000 张 Test 集做评测，验证动态扩缩容不会对模型收敛与准确率产生任何影响。

---

## 🛠️ 项目结构

```
gpud/
├── config.toml    # 声明式 GPU 配置文件 (活跃卡 active_gpus 定义)
├── gpud.py        # gpud 弹性调度装饰器与 Offload 管理器
├── demo.py        # ViT-H (224x224 CIFAR-10) 弹性训练 & 验证 Demo 脚本
├── prd.md         # 产品需求文档 (PRD)
├── README.md      # 英文 README
└── README.zh.md   # 中文 README
```

---

## 📄 许可证

[MIT License](LICENSE)
