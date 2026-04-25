from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def run(self, u):
        pass

    @abstractmethod
    def derivative(self, u):
        pass


class LinearFunction(ActivationFunction):
    def run(self, u: float) -> float:
        return u

    def derivative(self, u: float) -> float:
        return np.ones_like(u, dtype=float)


class SigmoidFunction(ActivationFunction):
    def run(self, u: float) -> float:
        return 1 / (1 + np.exp(-u))

    def derivative(self, u: float) -> float:
        sigmoid = self.run(u)
        return sigmoid * (1 - sigmoid)


class HyperbolicTangent(ActivationFunction):
    def __init__(self, a: float = 1, b: float = 1) -> None:
        self.a = a
        self.b = b

    def run(self, u: float) -> float:
        return self.a * np.tanh(self.b * u)

    def derivative(self, u: float) -> float:
        return self.a * self.b * (1 - np.tanh(self.b * u) ** 2)


class ReLUFunction(ActivationFunction):
    def run(self, u: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, u)

    def derivative(self, u: np.ndarray) -> np.ndarray:
        return (u > 0.0).astype(float)


class LeakyReLUFunction(ActivationFunction):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def run(self, u: np.ndarray) -> np.ndarray:
        return np.where(u > 0.0, u, self.alpha * u)

    def derivative(self, u: np.ndarray) -> np.ndarray:
        return np.where(u > 0.0, 1.0, self.alpha)


class ELUFunction(ActivationFunction):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def run(self, u: np.ndarray) -> np.ndarray:
        return np.where(u > 0.0, u, self.alpha * (np.exp(u) - 1.0))

    def derivative(self, u: np.ndarray) -> np.ndarray:
        return np.where(u > 0.0, 1.0, self.alpha * np.exp(u))


class SoftplusFunction(ActivationFunction):
    def run(self, u: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0.0)

    def derivative(self, u: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-u))


class SoftmaxFunction(ActivationFunction):
    def run(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u)
        if u.ndim == 1:
            shifted = u - np.max(u)
            e_u = np.exp(shifted)
            return e_u / np.sum(e_u)

        shifted = u - np.max(u, axis=1, keepdims=True)
        e_u = np.exp(shifted)
        return e_u / np.sum(e_u, axis=1, keepdims=True)

    def derivative(self, u: np.ndarray) -> np.ndarray:
        s = self.run(u)
        return s * (1 - s)
