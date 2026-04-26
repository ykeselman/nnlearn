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
        self.v = None

    def direction(self, grads: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros(grads.shape)
        self.v = self.beta * self.v + grads
        # alt with LR = 10*lr:
        # self.v = self.beta * self.v + (1 - self.beta) * grads
        return self.v


class RMSProp(Optimizer):
    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.G = None

    def direction(self, grads: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros(grads.shape)
        self.G = self.beta * self.G + (1-self.beta) * grads * grads
        return grads/(np.sqrt(self.G) + self.eps)


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
        self.v = None
        self.G = None
        self.t = 1

    def direction(self, grads: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)

        self.v = self.beta1 * self.v + (1-self.beta1) * grads
        self.G = self.beta2 * self.G + (1-self.beta2) * grads * grads

        v_hat = self.v / (1 - self.beta1**self.t)
        G_hat = self.G / (1 - self.beta2**self.t)

        self.t += 1

        return v_hat / (np.sqrt(G_hat) + self.eps)
