from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def run(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class MeanSquareError(LossFunction):
    def run(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class MeanAbsoluteError(LossFunction):
    def run(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def run(self, y_true, y_pred):
        error = y_pred - y_true
        abs_error = np.abs(error)
        quadratic = 0.5 * error**2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        return np.mean(np.where(abs_error <= self.delta, quadratic, linear))

    def derivative(self, y_true, y_pred):
        error = y_pred - y_true
        grad = np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error),
        )
        return grad / y_true.size


class BinaryCrossEntropy(LossFunction):
    def run(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        grad = (y_pred - y_true) / (y_pred * (1.0 - y_pred))
        return grad / y_true.size


class CrossEntropy(LossFunction):
    def run(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) / y_true.shape[0]
