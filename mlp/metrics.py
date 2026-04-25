from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    name = "metric"

    @abstractmethod
    def run(self, y_true, y_pred):
        pass


class MeanAbsoluteErrorMetric(Metric):
    name = "mae"

    def run(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))


class BinaryAccuracy(Metric):
    name = "binary_accuracy"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def run(self, y_true, y_pred):
        y_hat = (y_pred >= self.threshold).astype(int)
        return float(np.mean(y_hat == y_true))


class CategoricalAccuracy(Metric):
    name = "categorical_accuracy"

    def run(self, y_true, y_pred):
        y_hat = np.argmax(y_pred, axis=1)
        y_ref = np.argmax(y_true, axis=1)
        return float(np.mean(y_hat == y_ref))
