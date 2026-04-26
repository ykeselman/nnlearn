"""First-principles optimizers in NumPy.

All optimizers share the same update rule:

    params <- params - lr * direction(grads)

Each subclass only has to answer one question: given the current gradient
(and any state I've accumulated), which direction should I step? That keeps
the family tree readable — the differences between SGD, Momentum, RMSProp
and Adam collapse to four short `direction` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Base class for all optimizers.

    State (velocities, second moments, step counters) lives on the instance
    and is lazily initialized on the first `direction` call, so callers don't
    need to know the parameter shape up front.
    """

    lr: float

    @abstractmethod
    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Return the descent direction for this step. May update internal state."""
        ...

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.lr * self.direction(grads)


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Returns raw grads"""
        return grads


class Momentum(Optimizer):
    """Classical (Polyak) heavy-ball momentum."""

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v: np.ndarray | None = None

    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Returns running sum β·v + g"""
        if self.v is None:
            self.v = np.zeros_like(grads)
        self.v = self.beta * self.v + grads
        return self.v


class RMSProp(Optimizer):
    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s: np.ndarray | None = None

    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Returns g / (√s + ε) with EMA of g² """
        if self.s is None:
            self.s = np.zeros_like(grads)
        self.s = self.beta * self.s + (1 - self.beta) * grads * grads
        return grads / (np.sqrt(self.s) + self.eps)


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Returns bias-corrected m̂ / (√v̂ + ε)"""
        if self.m is None:
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads * grads
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return m_hat / (np.sqrt(v_hat) + self.eps)


class MomentumSign(Optimizer):
    """Soft-sign of an EMA of gradients (per-coordinate "adjusted direction").

    Direction is `m / (|m| + eps)`, applied componentwise. For |m| >> eps this
    is sign(m), so far from a minimum the update is axis-aligned with bounded
    step like Lion. As gradients vanish the denominator stops being dominated
    by |m| and the update collapses to `m / eps` — proportional to m, which
    drives step size to zero alongside the gradient. That's the same throttle
    Adam gets from `sqrt(v_hat) + eps`, and it removes the ±lr noise floor
    that pure `np.sign` left behind, with no lr schedule needed.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.m: np.ndarray | None = None

    def direction(self, grads: np.ndarray) -> np.ndarray:
        """Returns m / (|m| + eps), where m is an EMA of past gradients."""
        if self.m is None:
            self.m = np.zeros_like(grads)
        self.m = self.beta * self.m + (1 - self.beta) * grads
        return self.m / (np.abs(self.m) + self.eps)
