# gpud: Lightweight Declarative Elastic GPU Scaling & Memory Offloading Framework for PyTorch DDP

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文文档 (Chinese Readme)](README.zh.md)

`gpud` is an extremely lightweight, declarative elastic GPU scheduling and VRAM offloading framework designed for PyTorch Distributed Data Parallel (DDP).

During model training, you **do not need to terminate processes** or **reload checkpoints from disk**. By simply updating the `config.toml` configuration file, `gpud` dynamically resizes training GPU clusters and adjusts active card IDs (e.g. scaling from 8 GPUs `[0, 1, 2, 3, 4, 5, 6, 7]` to 4 GPUs `[0, 1, 2, 3]` or any arbitrary card subset) at Epoch boundaries, while completely freeing suspended GPU VRAM down to **0 MB**.

---

## 🌟 Key Features & Problem Solved

1. **Dynamic GPU Scaling (Expansion & Reduction)**:
   * Supports dynamic GPU count and card selection in Data Parallel (DDP) training (e.g., 8-card full cluster `[0, 1, 2, 3, 4, 5, 6, 7]`, 4-card scaling `[0, 1, 2, 3]`, or arbitrary subsets like `[0, 2, 5]`).
2. **Zero Downtime & Zero Checkpoint I/O Overhead**:
   * Eliminates the traditional high-latency workflow: *"kill process -> modify launch script -> reload large checkpoint from disk"*.
3. **Physical VRAM Offloading to 0 MB**:
   * Suspended GPU processes automatically transfer model parameters, buffers, and optimizer states (e.g., AdamW momentum) to CPU RAM, followed by `gc.collect()` and `torch.cuda.empty_cache()`, lowering GPU VRAM usage to **0 MB** for other users to borrow.
4. **Zero-Invasive Decorator (`@elastic_scheduler`)**:
   * Requires **only 1 line of decorator code** added to your single-epoch training function without cluttering user code with process group management or memory swapping.
5. **Adaptive Data Redistribution**:
   * Re-computes `DistributedSampler` replica allocations according to the active GPU count to guarantee dataset coverage and convergence math consistency.
6. **Optimized Multi-Worker CPU Data Loading**:
   * Automatically allocates **half of system CPU cores** (`os.cpu_count() // 2`) to DataLoader workers for optimal parallel data throughput.

---

## 🏗️ System Architecture

The architecture consists of three decoupled components:

```
┌─────────────────────────────────────────────────────────────┐
│                 Control Layer (config.toml)                 │
│         Declarative active_gpus = [0, 1, 2, 3]              │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Epoch-bound sync & broadcast)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Scheduler Layer (gpud.py)                   │
│        @elastic_scheduler (NCCL regrouping & Offload)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Injects ddp_model & dataloader)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Application Layer (demo.py / train.py)          │
│            Standard PyTorch Epoch Training Function         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation
`gpud` requires no heavy 3rd-party C++ extensions—only PyTorch (`torch`, `torchvision`) and the Python standard library.

```bash
git clone https://github.com/your-org/gpud.git
cd gpud
```

### 2. Configuration (`config.toml`)
Create or edit `config.toml` in your project root:

```toml
# Define active physical GPU ranks for the upcoming Epoch
active_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
```

### 3. Attach Decorator (1-Line Code Change)

```python
from gpud import elastic_scheduler

# ==============================================================================
# [GPUD DECORATOR]: Add only 1 line above your training function!
# ==============================================================================
@elastic_scheduler(config_path="config.toml")
def train_one_epoch(model, dataset, optimizer, criterion, epoch, device):
    model.train()
    for images, targets in dataset: # dataset is automatically replaced with elastic DataLoader
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, targets.to(device))
        loss.backward()
        optimizer.step()
```

### 4. Launch Distributed Training
Launch training via `torchrun` (e.g. on an 8-GPU node):

```bash
torchrun --nproc_per_node=8 demo.py
```

### 5. Live Scaling Walkthrough (Yield & Reclaim Cards)
During live training, modify `config.toml` in another terminal window:

* **Reduce GPUs (Yield cards to colleagues)**:
  Change to `active_gpus = [0, 1, 2, 3]` and save.
  *At the end of the current epoch, GPUs 4–7 offload model/optimizer states to CPU RAM and drop VRAM to 0 MB. GPUs 0–3 automatically handle 100% data partition and continue seamlessly.*

* **Expand GPUs (Reclaim cards for speed)**:
  Change to `active_gpus = [0, 1, 2, 3, 4, 5, 6, 7]` and save.
  *At the end of the current epoch, GPUs 4–7 reload state from CPU RAM to GPU VRAM, regroup NCCL, and accelerate training.*

---

## 💻 Demo Script Details (`demo.py`)

`demo.py` provides an out-of-the-box Vision Transformer distributed training example demonstrating all features of `gpud`:

* **Model Architecture (ViT-H)**: Vision Transformer Heavy (`img_size=224`, `patch_size=16`, `in_chans=3`, `embed_dim=1280`, `depth=32`, ~630 Million parameters). Substantial memory footprint allowing clear observation of 7.5GB+ per-GPU VRAM release down to **0 MB** when scaled down.
* **Dataset & Upsampling**: Full CIFAR-10 dataset (50,000 training images, 10,000 test images), upsampled to `224x224` resolution using bicubic interpolation (`transforms.InterpolationMode.BICUBIC`).
* **Infinite Epoch Training Loop**: Runs in an infinite `while True:` loop without maximum epoch limits, enabling continuous testing of live scaling via `config.toml`.
* **Classic `tqdm` Progress Bar**: Formatted with `ncols=150` on Rank 0, displaying real-time `train_loss`, `train_acc`, `val_loss`, `val_acc`, elapsed time, processing rate (it/s), and ETA.
* **Accuracy Validation**: Evaluates the 10,000 test images every epoch to prove dynamic GPU scaling does NOT impair model convergence or final accuracy.

---

## 🛠️ Repository Structure

```
gpud/
├── config.toml    # Declarative GPU configuration (active_gpus array)
├── gpud.py        # Elastic scheduler decorator & VRAM offload engine
├── demo.py        # ViT-H (224x224 CIFAR-10) elastic training & validation demo
├── prd.md         # Product Requirements Document
├── README.md      # English README
└── README.zh.md   # Chinese README
```

---

## 📄 License

[MIT License](LICENSE)
