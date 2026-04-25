import numpy as np

from ..activations import SoftmaxFunction
from ..core.layer import Layer
from ..losses import CrossEntropy
from ..optimizers import SGDOptimizer


class MultiLayerPerceptron:
    def __init__(self, x, y, learning_rate, loss_function, optimizer=None):
        self.x = self._ensure_2d(x)
        self.y = self._ensure_2d(y)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.optimizer = optimizer or SGDOptimizer(learning_rate)
        self.layers = []

    def _ensure_2d(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def add_layer(self, num_neurons, activation_function):
        num_inputs = self.x.shape[1] if len(self.layers) == 0 else len(self.layers[-1].neurons)
        self.layers.append(Layer(num_neurons, num_inputs, activation_function))

    def forward_pass(self, inputs):
        if not self.layers:
            raise ValueError("At least one layer must be added before forward pass.")
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward_pass(self, y_pred, y_true):
        use_softmax_ce = (
            isinstance(self.loss_function, CrossEntropy)
            and isinstance(self.layers[-1].activation_function, SoftmaxFunction)
        )

        if use_softmax_ce:
            delta = (y_pred - y_true) / y_true.shape[0]
            delta = self.layers[-1].backward(
                delta,
                optimizer=self.optimizer,
                apply_activation_derivative=False,
            )
            layers = reversed(self.layers[:-1])
        else:
            delta = self.loss_function.derivative(y_true, y_pred)
            layers = reversed(self.layers)

        for layer in layers:
            delta = layer.backward(delta, optimizer=self.optimizer)

    def _run_metric(self, metric, y_true, y_pred):
        if hasattr(metric, "run"):
            name = getattr(metric, "name", metric.__class__.__name__.lower())
            return name, float(metric.run(y_true, y_pred))

        if callable(metric):
            name = getattr(metric, "__name__", "custom_metric")
            return name, float(metric(y_true, y_pred))

        raise ValueError("Each metric must be a callable or provide a .run method.")

    def evaluate(self, x, y, metrics=None):
        y_eval = self._ensure_2d(y)
        y_pred = self.predict(x)
        results = {"loss": float(self.loss_function.run(y_eval, y_pred))}

        if metrics:
            for metric in metrics:
                name, value = self._run_metric(metric, y_eval, y_pred)
                results[name] = value

        return results

    def train(self, epochs, verbose=True, batch_size=None, shuffle=True, metrics=None):
        history = {"loss": []}
        if metrics:
            for metric in metrics:
                name, _ = self._run_metric(metric, self.y, self.forward_pass(self.x))
                history[name] = []

        if batch_size is None:
            batch_size = self.x.shape[0]

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        n_samples = self.x.shape[0]

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                x_epoch = self.x[indices]
                y_epoch = self.y[indices]
            else:
                x_epoch = self.x
                y_epoch = self.y

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                x_batch = x_epoch[start:end]
                y_batch = y_epoch[start:end]
                y_pred_batch = self.forward_pass(x_batch)
                self.backward_pass(y_pred_batch, y_batch)

            y_pred = self.forward_pass(self.x)
            loss = float(self.loss_function.run(self.y, y_pred))
            history["loss"].append(loss)

            metrics_log = ""
            if metrics:
                parts = []
                for metric in metrics:
                    name, value = self._run_metric(metric, self.y, y_pred)
                    history[name].append(value)
                    parts.append(f"{name}: {value:.6f}")
                metrics_log = " - " + " - ".join(parts)

            if verbose:
                print(f"Epoch {epoch + 1}, Loss: {loss}{metrics_log}")

        return history

    def predict(self, x):
        return self.forward_pass(self._ensure_2d(x))
