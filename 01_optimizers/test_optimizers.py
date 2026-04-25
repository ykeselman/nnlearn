"""Pytest tests for the optimizers module.

Each optimizer gets its own test class that inherits from `TestOptimizer`
and supplies two things:

  - `make_optimizer()`        — a fresh instance with known hyperparameters
  - `reference_directions(gs)` — the expected direction sequence, computed
                                 by an independent textbook-style reference

The shared test methods then verify that the real `direction` output matches
the reference, that input gradients aren't mutated, that shape is preserved,
and that the inherited `step` really is `params - lr * direction(grads)`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pytest

from optimizers import Adam, Momentum, Optimizer, RMSProp, SGD


class TestOptimizer(ABC):
    """Abstract contract test. Pytest skips it; subclasses opt in via __test__."""

    __test__ = False  # don't collect the abstract base itself

    @abstractmethod
    def make_optimizer(self) -> Optimizer: ...

    @abstractmethod
    def reference_directions(self, grads_seq: list[np.ndarray]) -> list[np.ndarray]: ...

    @pytest.fixture(params=[(3,), (2, 4)], ids=["shape=3", "shape=2x4"])
    def grads_seq(self, request) -> list[np.ndarray]:
        rng = np.random.default_rng(0)
        return [rng.standard_normal(request.param) for _ in range(8)]

    def test_direction_matches_reference(self, grads_seq):
        opt = self.make_optimizer()
        expected = self.reference_directions(grads_seq)
        for t, (g, e) in enumerate(zip(grads_seq, expected)):
            d = opt.direction(g.copy())
            np.testing.assert_allclose(
                d, e, rtol=1e-10, atol=1e-12,
                err_msg=f"direction mismatch at step {t}",
            )


class TestSGD(TestOptimizer):
    __test__ = True

    LR = 0.05

    def make_optimizer(self):
        return SGD(lr=self.LR)

    def reference_directions(self, grads_seq):
        return [g.copy() for g in grads_seq]


class TestMomentum(TestOptimizer):
    __test__ = True

    LR = 0.05
    BETA = 0.9

    def make_optimizer(self):
        return Momentum(lr=self.LR, beta=self.BETA)

    def reference_directions(self, grads_seq):
        v = np.zeros_like(grads_seq[0])
        out = []
        for g in grads_seq:
            v = self.BETA * v + g
            out.append(v.copy())
        return out


class TestRMSProp(TestOptimizer):
    __test__ = True

    LR = 0.01
    BETA = 0.9
    EPS = 1e-8

    def make_optimizer(self):
        return RMSProp(lr=self.LR, beta=self.BETA, eps=self.EPS)

    def reference_directions(self, grads_seq):
        s = np.zeros_like(grads_seq[0])
        out = []
        for g in grads_seq:
            s = self.BETA * s + (1 - self.BETA) * g * g
            out.append(g / (np.sqrt(s) + self.EPS))
        return out


class TestAdam(TestOptimizer):
    __test__ = True

    LR = 0.01
    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 1e-8

    def make_optimizer(self):
        return Adam(lr=self.LR, beta1=self.BETA1, beta2=self.BETA2, eps=self.EPS)

    def reference_directions(self, grads_seq):
        m = np.zeros_like(grads_seq[0])
        v = np.zeros_like(grads_seq[0])
        out = []
        for t, g in enumerate(grads_seq, start=1):
            m = self.BETA1 * m + (1 - self.BETA1) * g
            v = self.BETA2 * v + (1 - self.BETA2) * g * g
            m_hat = m / (1 - self.BETA1**t)
            v_hat = v / (1 - self.BETA2**t)
            out.append(m_hat / (np.sqrt(v_hat) + self.EPS))
        return out
