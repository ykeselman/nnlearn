"""Deep MLP on MNIST: Xavier vs Kaiming init, with vs without BatchNorm.

Two effects we're trying to make visible:

1. *Vanishing gradients under bad init.* A stack of ReLU linears initialized
   with Xavier (which assumes a tanh-like activation with unit derivative at
   zero) under-scales weights by sqrt(2). Over a 15-layer net, that's roughly
   2^7.5 ~ 180x of variance shrinkage, and the same compounding hits gradients
   on the way back. Kaiming corrects for ReLU. BatchNorm renormalizes
   pre-activations every step and largely papers over a bad init.

2. *BN's generalization advantage.* The optimization smoothing + per-step
   batch-statistic noise act as mild regularization. To make this visible we
   train on a small subset (5000 examples) so the no-BN runs can overfit, and
   evaluate on the full MNIST test set every epoch.

Outputs (alongside this script):
  - loss_curves.png   train loss / test loss / test accuracy for the 4 configs
  - grad_norms.png    per-layer gradient L2 norm: line plot at init + 2x2 heatmap
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


HIDDEN = 128
N_LINEAR = 15           # total Linear layers (14 hidden + 1 classifier)
INPUT = 28 * 28
OUTPUT = 10
BATCH_SIZE = 128
EPOCHS = 30
TRAIN_SUBSET = 5000     # small enough that no-BN runs can overfit
# Plain SGD (no momentum, no adaptive scaling) so vanishing/exploding gradients
# actually show up in the loss. Adam would silently rescale them away.
LR = 0.05
SEED = 0
# MNIST cached in a sibling project — reuse to avoid re-downloading.
MNIST_ROOT = "/home/yakov/Studies/Fawaz-NN-Bootcamp/data"


def init_linear(layer: nn.Linear, scheme: str) -> None:
    if scheme == "xavier":
        nn.init.xavier_uniform_(layer.weight, gain=1.0)
    elif scheme == "kaiming":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(scheme)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(self, n_linear: int, hidden: int, init_scheme: str, batch_norm: bool):
        super().__init__()
        modules: list[nn.Module] = []
        in_dim = INPUT
        for _ in range(n_linear - 1):
            lin = nn.Linear(in_dim, hidden)
            init_linear(lin, init_scheme)
            modules.append(lin)
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden))
            modules.append(nn.ReLU())
            in_dim = hidden
        out = nn.Linear(in_dim, OUTPUT)
        init_linear(out, init_scheme)
        modules.append(out)
        self.net = nn.Sequential(*modules)
        # Stable references for grad-norm logging, in input-to-output order.
        self.linears = [m for m in self.net if isinstance(m, nn.Linear)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.numel()
    model.train()
    return total_loss / n, correct / n


def train_one(
    init_scheme: str,
    batch_norm: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    torch.manual_seed(SEED)
    model = MLP(N_LINEAR, HIDDEN, init_scheme, batch_norm).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    train_losses: list[float] = []
    grad_norms: list[list[float]] = []   # (steps, n_linear)
    test_losses: list[float] = []
    test_accs: list[float] = []

    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            grad_norms.append([lin.weight.grad.norm().item() for lin in model.linears])
            train_losses.append(loss.item())
            opt.step()

        tl, ta = evaluate(model, test_loader, device)
        test_losses.append(tl)
        test_accs.append(ta)
        print(f"  epoch {epoch+1:2d}/{EPOCHS}: test_loss={tl:.4f}  test_acc={ta*100:.2f}%")

    return {
        "train_loss": np.array(train_losses),
        "grad_norms": np.array(grad_norms),
        "test_loss": np.array(test_losses),
        "test_acc": np.array(test_accs),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = Path(__file__).resolve().parent

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(root=MNIST_ROOT, train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(root=MNIST_ROOT, train=False, download=False, transform=transform)

    # Deterministic subset so all four configs see exactly the same training set.
    rng = np.random.default_rng(SEED)
    subset_idx = rng.choice(len(full_train), size=TRAIN_SUBSET, replace=False)
    train_ds = Subset(full_train, subset_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=0,
    )

    configs = [
        ("Xavier, no BN",  "xavier",  False, "tab:blue"),
        ("Xavier, BN",     "xavier",  True,  "tab:cyan"),
        ("Kaiming, no BN", "kaiming", False, "tab:red"),
        ("Kaiming, BN",    "kaiming", True,  "tab:orange"),
    ]
    results: dict[str, dict] = {}
    colors: dict[str, str] = {}
    for name, scheme, bn, color in configs:
        print(f"\ntraining {name} ...")
        results[name] = train_one(scheme, bn, train_loader, test_loader, device)
        colors[name] = color
        print(
            f"  final train_loss={results[name]['train_loss'][-50:].mean():.4f}  "
            f"final test_loss={results[name]['test_loss'][-1]:.4f}  "
            f"best test_acc={results[name]['test_acc'].max()*100:.2f}%"
        )

    # --- loss curves: train (per step), test loss, test accuracy ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    kernel = 30
    for name, color in colors.items():
        train_loss = results[name]["train_loss"]
        smoothed = np.convolve(train_loss, np.ones(kernel) / kernel, mode="valid")
        axes[0].plot(smoothed, color=color, lw=1.4, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel(f"train loss (smoothed, window={kernel})")
    axes[0].set_title("Training loss")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    epochs_x = np.arange(1, EPOCHS + 1)
    for name, color in colors.items():
        axes[1].plot(epochs_x, results[name]["test_loss"], "o-",
                     color=color, lw=1.4, ms=4, label=name)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test loss")
    axes[1].set_title("Test loss (generalization)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    for name, color in colors.items():
        axes[2].plot(epochs_x, results[name]["test_acc"] * 100, "o-",
                     color=color, lw=1.4, ms=4, label=name)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("test accuracy (%)")
    axes[2].set_title("Test accuracy")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="lower right")

    fig.suptitle(
        f"{N_LINEAR}-layer MLP on MNIST  "
        f"(train subset = {TRAIN_SUBSET}, SGD lr={LR})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(here / "loss_curves.png", dpi=150, bbox_inches="tight")

    # --- per-layer gradient norms ---
    # Top: line plot at init (mean over first 10 steps to denoise a bit).
    # Bottom (2x2): heatmap of log10 grad norm per layer over training.
    fig2 = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.45, wspace=0.18)

    ax_top = fig2.add_subplot(gs[0, :])
    layers = np.arange(1, N_LINEAR + 1)
    for name, color in colors.items():
        init_mean = results[name]["grad_norms"][:10].mean(axis=0)
        ax_top.plot(layers, init_mean, "o-", color=color, lw=1.6, label=name)
    ax_top.set_yscale("log")
    ax_top.set_xticks(layers)
    ax_top.set_xlabel("layer index (1 = closest to input)")
    ax_top.set_ylabel("weight grad L2 norm")
    ax_top.set_title("Gradient norm per layer at initialization (mean of first 10 steps)")
    ax_top.grid(True, which="both", alpha=0.3)
    ax_top.legend(loc="best")

    # Common color range across all four heatmaps (log scale). Clip to the
    # smallest observed positive value so the colormap spans the full range
    # of actually-occurring norms.
    all_g = np.concatenate([results[n]["grad_norms"].ravel() for n in colors])
    floor = all_g[all_g > 0].min()
    vmin, vmax = np.log10(floor), np.log10(all_g.max())

    heatmap_axes = [fig2.add_subplot(gs[1 + i // 2, i % 2]) for i in range(4)]
    im = None
    for ax_h, name in zip(heatmap_axes, colors):
        gnorms = results[name]["grad_norms"]
        im = ax_h.imshow(
            np.log10(np.maximum(gnorms.T, floor)),
            aspect="auto", origin="lower",
            vmin=vmin, vmax=vmax, cmap="viridis",
            extent=[0, gnorms.shape[0], 0.5, N_LINEAR + 0.5],
        )
        ax_h.set_title(name)
        ax_h.set_xlabel("step")
        ax_h.set_ylabel("layer")
        ax_h.set_yticks(layers[::2])

    cbar = fig2.colorbar(
        im, ax=heatmap_axes, orientation="vertical",
        fraction=0.025, pad=0.02, shrink=0.95,
    )
    cbar.set_label("log10  ||dL/dW||₂")

    fig2.suptitle("Per-layer weight gradient norms — vanishing/exploding under different inits", y=0.995)
    fig2.savefig(here / "grad_norms.png", dpi=150, bbox_inches="tight")

    print(f"\nwrote {here/'loss_curves.png'} and {here/'grad_norms.png'}")


if __name__ == "__main__":
    main()
