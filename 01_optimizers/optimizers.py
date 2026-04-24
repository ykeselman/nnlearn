"""First-principles optimizers in NumPy.

Each optimizer holds its own state and exposes a single `step(params, grads)`
method that returns the next parameter vector. Nothing fancy — the goal is
clarity, not performance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Base class for all optimizers.

    An optimizer is a pure state-holding object with one job: given the current
    `params` and their `grads`, return the updated params. Any running state
    (velocities, second moments, step counters) lives on the instance and is
    lazily initialized on the first `step` call so callers don't need to know
    the parameter shape up front.
    """

    lr: float

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Return the next parameter vector. Must not mutate `params`."""
        ...


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        return params - self.lr * grads


class Momentum(Optimizer):
    """Classical (Polyak) heavy-ball momentum."""

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


class RMSProp(Optimizer):
    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(params)
        self.s = self.beta * self.s + (1 - self.beta) * grads * grads
        return params - self.lr * grads / (np.sqrt(self.s) + self.eps)


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

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads * grads
        # Bias correction.
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
