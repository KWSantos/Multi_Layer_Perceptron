import unittest
import numpy as np

from mlp.activations import (
    ELUFunction,
    LeakyReLUFunction,
    LinearFunction,
    ReLUFunction,
    SigmoidFunction,
    SoftplusFunction,
)
from mlp.core.layer import Layer
from mlp.losses import (
    BinaryCrossEntropy,
    CrossEntropy,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquareError,
)
from mlp.metrics import BinaryAccuracy, MeanAbsoluteErrorMetric
from mlp.models.mlp import MultiLayerPerceptron
from mlp.optimizers import AdamOptimizer, MomentumOptimizer


class MLPTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_mse_contract_and_shape(self):
        y_true = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.8], [0.2]])
        mse = MeanSquareError()

        loss = mse.run(y_true, y_pred)
        grad = mse.derivative(y_true, y_pred)

        self.assertIsInstance(loss, float)
        self.assertEqual(grad.shape, y_true.shape)

    def test_new_activation_functions_have_consistent_shapes(self):
        x = np.array([[-1.0, 0.0, 2.0]])
        activations = [
            ReLUFunction(),
            LeakyReLUFunction(alpha=0.1),
            ELUFunction(alpha=1.0),
            SoftplusFunction(),
        ]

        for activation in activations:
            y = activation.run(x)
            dy = activation.derivative(x)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(dy.shape, x.shape)

    def test_new_loss_functions_contract_and_shape(self):
        y_true = np.array([[1.0], [0.0], [1.0]])
        y_pred = np.array([[0.8], [0.2], [0.7]])
        losses = [
            MeanAbsoluteError(),
            HuberLoss(delta=1.0),
            BinaryCrossEntropy(),
            CrossEntropy(),
        ]

        y_true_ce = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred_ce = np.array([[0.9, 0.1], [0.2, 0.8]])

        for loss_fn in losses:
            if isinstance(loss_fn, CrossEntropy):
                loss = loss_fn.run(y_true_ce, y_pred_ce)
                grad = loss_fn.derivative(y_true_ce, y_pred_ce)
                self.assertEqual(grad.shape, y_true_ce.shape)
            else:
                loss = loss_fn.run(y_true, y_pred)
                grad = loss_fn.derivative(y_true, y_pred)
                self.assertEqual(grad.shape, y_true.shape)
            self.assertIsInstance(loss, float)

    def test_layer_backward_returns_previous_delta_shape(self):
        layer = Layer(num_neurons=3, num_inputs=2, activation_function=SigmoidFunction())
        x = np.random.randn(5, 2)
        out = layer.forward(x)
        delta = np.ones_like(out)

        delta_prev = layer.backward(delta, learning_rate=0.1)

        self.assertEqual(delta_prev.shape, x.shape)

    def test_training_reduces_loss_in_simple_regression(self):
        x = np.random.randn(50, 2)
        y = 0.7 * x[:, [0]] - 0.2 * x[:, [1]] + 0.1

        model = MultiLayerPerceptron(
            x=x,
            y=y,
            learning_rate=0.1,
            loss_function=MeanSquareError(),
        )
        model.add_layer(num_neurons=1, activation_function=LinearFunction())

        initial_pred = model.predict(x)
        initial_loss = model.loss_function.run(y, initial_pred)

        model.train(epochs=200, verbose=False)

        final_pred = model.predict(x)
        final_loss = model.loss_function.run(y, final_pred)

        self.assertLess(final_loss, initial_loss)

    def test_training_with_minibatch_metrics_and_adam(self):
        x = np.random.randn(80, 2)
        y = 0.4 * x[:, [0]] + 0.6 * x[:, [1]] - 0.3

        model = MultiLayerPerceptron(
            x=x,
            y=y,
            learning_rate=0.05,
            loss_function=MeanSquareError(),
            optimizer=AdamOptimizer(learning_rate=0.05),
        )
        model.add_layer(num_neurons=4, activation_function=ReLUFunction())
        model.add_layer(num_neurons=1, activation_function=LinearFunction())

        history = model.train(
            epochs=60,
            verbose=False,
            batch_size=16,
            shuffle=True,
            metrics=[MeanAbsoluteErrorMetric()],
        )
        self.assertIn("loss", history)
        self.assertIn("mae", history)
        self.assertEqual(len(history["loss"]), 60)
        self.assertEqual(len(history["mae"]), 60)
        self.assertLess(history["loss"][-1], history["loss"][0])

    def test_evaluate_with_binary_accuracy(self):
        x = np.random.randn(60, 2)
        y = (x[:, [0]] + x[:, [1]] > 0).astype(float)

        model = MultiLayerPerceptron(
            x=x,
            y=y,
            learning_rate=0.1,
            loss_function=BinaryCrossEntropy(),
            optimizer=MomentumOptimizer(learning_rate=0.1, momentum=0.8),
        )
        model.add_layer(num_neurons=1, activation_function=SigmoidFunction())
        model.train(epochs=100, verbose=False, batch_size=12)

        result = model.evaluate(x, y, metrics=[BinaryAccuracy()])
        self.assertIn("loss", result)
        self.assertIn("binary_accuracy", result)


if __name__ == "__main__":
    unittest.main()
