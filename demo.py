"""
==============================================================================
demo.py - 基于 Vision Transformer (ViT-H) 与 CIFAR-10 (224x224) 的 gpud 弹性训练示例
==============================================================================
说明：
本脚本演示如何在常规 PyTorch 分布式数据并行 (DDP) 训练代码中，仅通过添加【1-2 行代码】接入 gpud 调度器。
在训练过程中，在终端修改 config.toml 中的 active_gpus 即可实现零中断的动态增容/减容与显存换入换出 (Offloading)。

服务端多卡运行指令（例如 8 卡服务器）：
    torchrun --nproc_per_node=8 demo.py

本地单机调试指令：
    python demo.py
==============================================================================
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# ==============================================================================
# 【GPUD 调度器导入】：仅需引入 elastic_scheduler 装饰器 (添加 1 行)
# ==============================================================================
from gpud import elastic_scheduler


# ------------------------------------------------------------------------------
# 1. 建模部分：ViT-H (Vision Transformer Heavy) 架构定义
# ------------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """图像 Patch 切分与 Embedding 映射层"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1280):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder 块 (Multi-Head Attention + MLP)"""
    def __init__(self, embed_dim=1280, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerH(nn.Module):
    """
    ViT-H (Huge) 架构定义：
    输入尺寸 224x224，Patch 16x16，3 通道，包含 32 层 Transformer Block，1280 维隐藏特征向量，约 6.3 亿参数。
    当卡 4~7 被动态缩容让出时，能直观看到每张卡 7.5GB+ 的物理显存完全释放为 0 MB。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=1280, depth=32, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ------------------------------------------------------------------------------
# 2. 数据准备：CIFAR-10 数据集 (全量 50k 训练 / 10k 测试，双三次插值至 224x224)
# ------------------------------------------------------------------------------
class SyntheticCIFAR10Dataset(Dataset):
    """用于无网络连接/离线自动测试时的伪 CIFAR-10 数据集 (224x224, 3通道)"""
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 3, 224, 224)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def get_cifar10_datasets():
    """获取 CIFAR-10 训练集与测试集（以 Bicubic 双三次插值调整为 224x224）"""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
    except Exception:
        print("[Demo Notice] 网络无法直接下载 CIFAR-10 数据集，自动切换为离线合成 CIFAR-10 模式。")
        train_dataset = SyntheticCIFAR10Dataset(size=2000)
        test_dataset = SyntheticCIFAR10Dataset(size=500)
    return train_dataset, test_dataset


# ------------------------------------------------------------------------------
# 3. Epoch 训练与验证逻辑
# ------------------------------------------------------------------------------
# ==============================================================================
# 【GPUD 弹性调度装饰器接入】：仅需加此 1 行装饰器！即刻接管多卡扩缩容与显存 Offload
# ==============================================================================
@elastic_scheduler(config_path="config.toml")
def train_one_epoch(model, dataset, optimizer, criterion, epoch, device, last_val_loss=None, last_val_acc=None):
    """
    单 Epoch 训练函数：
    在 @elastic_scheduler 装饰的作用下：
    1. model 在活跃卡上被自动包装为 DistributedDataParallel (DDP)。
    2. dataset 被依据 active_gpus 列表重新切分，并分配 CPU 核心数一半的 num_workers 并行加载。
    3. 落选（缩容）的 GPU 进程在此函数处自动暂停计算，模型与优化器状态自动被 Offload 至 CPU RAM，显存清 0 MB。
    """
    is_cuda = next(model.parameters()).is_cuda
    current_device = next(model.parameters()).device if is_cuda else torch.device("cpu")

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    rank = int(os.environ.get("RANK", 0))
    is_rank0 = (rank == 0)

    # 这里的 dataset 已被 gpud 装饰器替换为分配好子卡 Sampler 的 DataLoader
    dataloader = dataset

    # 使用经典 tqdm 风格进度条 (包含 进度%, Step, 耗时, ETA, 速率, train_loss, train_acc, val_loss, val_acc)
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:04d} | Train",
        disable=not is_rank0,
        leave=True,
        dynamic_ncols=True
    )

    for images, targets in pbar:
        images = images.to(current_device)
        targets = targets.to(current_device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if is_rank0:
            postfix = {
                "train_loss": f"{loss.item():.4f}",
                "train_acc": f"{100.0 * correct / max(1, total):.2f}%"
            }
            if last_val_loss is not None and last_val_acc is not None:
                postfix["val_loss"] = f"{last_val_loss:.4f}"
                postfix["val_acc"] = f"{last_val_acc:.2f}%"
            pbar.set_postfix(postfix)

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)

    if is_rank0:
        print(f"--> [Epoch {epoch:04d}] Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}% | 耗时: {elapsed:.2f}s")
    return avg_loss, accuracy


def validate(model, val_dataset, criterion, epoch, device):
    """
    模型评估验证逻辑：使用 CIFAR-10 全量 Test 集评测模型准确率 (Acc)，验证动态调度不会对模型收敛产生任何影响
    """
    is_cuda = next(model.parameters()).is_cuda
    current_device = next(model.parameters()).device if is_cuda else torch.device("cpu")
    rank = int(os.environ.get("RANK", 0))
    is_rank0 = (rank == 0)

    val_loader = val_dataset if isinstance(val_dataset, DataLoader) else DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=max(1, (os.cpu_count() or 2) // 2)
    )

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:04d} | Val  ",
            disable=not is_rank0,
            leave=False,
            dynamic_ncols=True
        )
        for images, targets in pbar:
            images, targets = images.to(current_device), targets.to(current_device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if is_rank0:
                pbar.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_acc": f"{100.0 * correct / max(1, total):.2f}%"
                })

    avg_loss = total_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)

    if is_rank0:
        print(f"==> [Epoch {epoch:04d}] Val Loss: {avg_loss:.4f} | Val Acc: {accuracy:.2f}%\n")
    return avg_loss, accuracy


# ------------------------------------------------------------------------------
# 4. 主程序入口 (无限循环无限 Epoch 训练)
# ------------------------------------------------------------------------------
def main():
    print("==================================================================")
    print("     gpud ViT-H CIFAR-10 (224x224) 弹性调度 & 验证 Demo          ")
    print("==================================================================")

    half_cpus = max(1, (os.cpu_count() or 2) // 2)
    print(f"[系统环境] CPU 核心总数: {os.cpu_count()}，gpud DataLoader 将使用 {half_cpus} 线程加载。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 实例化 ViT-H (224x224, 3通道, patch_size=16) 大模型
    print("[初始化] 正在构建 ViT-H (224x224 3通道, ~6.3 亿参数) 大模型...")
    model = VisionTransformerH(
        img_size=224, patch_size=16, in_chans=3, num_classes=10,
        embed_dim=1280, depth=32, num_heads=16
    ).to(device)

    # 2. 获取 CIFAR-10 数据集 (全量 50k 训练集 + 10k 测试集)
    train_dataset, test_dataset = get_cifar10_datasets()

    # 3. 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # 4. 无限循环训练 (无最大 Epoch 限制)
    print("\n[开始无限循环训练] 没有最大 Epoch 限制。使用 Ctrl+C 中断。")
    print("------------------------------------------------------------------")
    print("提示：训练过程中，你可以随时打开并编辑 config.toml 文件，例如：")
    print("  - 全量 8 卡： active_gpus = [0, 1, 2, 3, 4, 5, 6, 7]")
    print("  - 动态减容至 4 卡： active_gpus = [0, 1, 2, 3]")
    print("  - 任意卡号组合： active_gpus = [0, 2, 5]")
    print("------------------------------------------------------------------\n")

    epoch = 1
    val_loss = None
    val_acc = None
    while True:
        train_loss, train_acc = train_one_epoch(
            model, train_dataset, optimizer, criterion, epoch, device,
            last_val_loss=val_loss, last_val_acc=val_acc
        )
        val_loss, val_acc = validate(model, test_dataset, criterion, epoch, device)
        epoch += 1


if __name__ == "__main__":
    main()
