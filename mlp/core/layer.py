import numpy as np

from .neuron import Neuron


class Layer:
    _counter = 0

    def __init__(self, num_neurons, num_inputs, activation_function):
        self.layer_id = Layer._counter
        Layer._counter += 1
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.activation_function = activation_function

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.array([neuron.get_sum(inputs) for neuron in self.neurons]).T
        self.a = self.activation_function.run(self.z)
        return self.a

    def backward(
        self,
        delta,
        learning_rate=None,
        optimizer=None,
        apply_activation_derivative=True,
    ):
        if learning_rate is None and optimizer is None:
            raise ValueError("Either learning_rate or optimizer must be provided.")

        delta_prev = np.zeros_like(self.inputs, dtype=float)
        batch_size = self.inputs.shape[0]

        for i, neuron in enumerate(self.neurons):
            if apply_activation_derivative:
                dz = delta[:, i] * self.activation_function.derivative(self.z[:, i])
            else:
                dz = delta[:, i]
            dz = dz.reshape(-1, 1)

            old_weights = neuron.weights.copy()
            grad_w = np.sum(dz * self.inputs, axis=0) / batch_size
            grad_b = np.sum(dz) / batch_size

            delta_prev += dz * old_weights

            if optimizer is not None:
                neuron.weights = optimizer.update(
                    f"layer{self.layer_id}_neuron{i}_weights",
                    neuron.weights,
                    grad_w,
                )
                neuron.bias = optimizer.update(
                    f"layer{self.layer_id}_neuron{i}_bias",
                    neuron.bias,
                    grad_b,
                )
            else:
                neuron.weights -= learning_rate * grad_w
                neuron.bias -= learning_rate * grad_b

        return delta_prev
