"""Benchmark SGD / Momentum / RMSProp / Adam on the Rosenbrock function.

Rosenbrock: f(x, y) = (a - x)^2 + b * (y - x^2)^2, with a=1, b=100.
Global minimum at (1, 1) with f = 0. The function has a long, narrow,
banana-shaped valley — easy to reach the valley, hard to move along it.
That's why it's a standard test for first-order optimizers.

Two plots are written alongside this script:
  - trajectories.png  contour map + each optimizer's path
  - loss_curves.png   f(x_t) vs iteration, log scale
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from optimizers import SGD, Momentum, RMSProp, Adam


A, B = 1.0, 100.0


def rosenbrock(p: np.ndarray) -> float:
    x, y = p
    return (A - x) ** 2 + B * (y - x * x) ** 2


def rosenbrock_grad(p: np.ndarray) -> np.ndarray:
    x, y = p
    dx = -2 * (A - x) - 4 * B * x * (y - x * x)
    dy = 2 * B * (y - x * x)
    return np.array([dx, dy])


def run(optimizer, start: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (trajectory, losses) with shapes (n_steps+1, 2) and (n_steps+1,)."""
    p = start.copy()
    traj = [p.copy()]
    losses = [rosenbrock(p)]
    for _ in range(n_steps):
        g = rosenbrock_grad(p)
        p = optimizer.step(p, g)
        traj.append(p.copy())
        losses.append(rosenbrock(p))
    return np.array(traj), np.array(losses)


def main() -> None:
    start = np.array([-1.5, 2.0])
    n_steps = 5000

    # Per-optimizer learning rates: plain SGD diverges on Rosenbrock at the
    # lr Adam prefers, and vice versa. These are hand-tuned to give each
    # method a fair shot from the same starting point.
    runs = {
        "SGD":      (SGD(lr=1e-3),                        "tab:blue"),
        "Momentum": (Momentum(lr=1e-3, beta=0.9),         "tab:orange"),
        "RMSProp":  (RMSProp(lr=1e-2, beta=0.9),          "tab:green"),
        "Adam":     (Adam(lr=1e-2, beta1=0.9, beta2=0.999), "tab:red"),
    }

    results = {}
    for name, (opt, color) in runs.items():
        traj, losses = run(opt, start, n_steps)
        results[name] = (traj, losses, color)
        final = traj[-1]
        print(
            f"{name:>9s}: final=({final[0]:+.4f}, {final[1]:+.4f})  "
            f"f={losses[-1]:.3e}  min_f={losses.min():.3e}"
        )

    # --- contour + trajectories ---
    xs = np.linspace(-2.0, 2.0, 400)
    ys = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (A - X) ** 2 + B * (Y - X * X) ** 2

    fig, ax = plt.subplots(figsize=(9, 7))
    # Log-spaced contours since Rosenbrock spans many orders of magnitude.
    levels = np.logspace(-1, 3.5, 25)
    ax.contour(X, Y, Z, levels=levels, norm="log", cmap="Greys", linewidths=0.6)
    for name, (traj, _, color) in results.items():
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.4, label=name, alpha=0.9)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, zorder=5)
    ax.scatter(*start, marker="o", color="black", s=40, label="start", zorder=6)
    ax.scatter(1.0, 1.0, marker="*", color="gold", s=180,
               edgecolors="black", linewidths=0.8, label="minimum (1,1)", zorder=6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Optimizer trajectories on Rosenbrock ({n_steps} steps)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("trajectories.png", dpi=150)

    # --- loss curves ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for name, (_, losses, color) in results.items():
        ax2.plot(losses, color=color, lw=1.2, label=name)
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("f(x, y)  (log scale)")
    ax2.set_title("Convergence on Rosenbrock")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("loss_curves.png", dpi=150)

    print("\nwrote trajectories.png and loss_curves.png")


if __name__ == "__main__":
    main()
