from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, param_id, param, grad):
        pass


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, param_id, param, grad):
        del param_id
        return param - self.learning_rate * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, param_id, param, grad):
        v = self.velocity.get(param_id, np.zeros_like(param))
        v = self.momentum * v - self.learning_rate * grad
        self.velocity[param_id] = v
        return param + v


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param_id, param, grad):
        m = self.m.get(param_id, np.zeros_like(param))
        v = self.v.get(param_id, np.zeros_like(param))
        t = self.t.get(param_id, 0) + 1

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad**2)

        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        updated = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        self.m[param_id] = m
        self.v[param_id] = v
        self.t[param_id] = t
        return updated
