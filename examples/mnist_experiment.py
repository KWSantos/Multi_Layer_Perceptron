import argparse
import os
import struct

import numpy as np

from mlp import (
    AdamOptimizer,
    CategoricalAccuracy,
    CrossEntropy,
    MultiLayerPerceptron,
    ReLUFunction,
    SoftmaxFunction,
)


def find_mnist_idx_files(root_path):
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    for root, _, files in os.walk(root_path):
        lower_files = {f.lower(): f for f in files}
        if "train-images.idx3-ubyte" in lower_files:
            train_images = os.path.join(root, lower_files["train-images.idx3-ubyte"])
        if "train-labels.idx1-ubyte" in lower_files:
            train_labels = os.path.join(root, lower_files["train-labels.idx1-ubyte"])
        if "t10k-images.idx3-ubyte" in lower_files:
            test_images = os.path.join(root, lower_files["t10k-images.idx3-ubyte"])
        if "t10k-labels.idx1-ubyte" in lower_files:
            test_labels = os.path.join(root, lower_files["t10k-labels.idx1-ubyte"])
    return train_images, train_labels, test_images, test_labels


def one_hot(labels, num_classes=10):
    return np.eye(num_classes, dtype=float)[labels]


def read_idx_images(file_path):
    with open(file_path, "rb") as f:
        magic, n_items, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {file_path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_items, rows * cols)
    return data


def read_idx_labels(file_path):
    with open(file_path, "rb") as f:
        magic, n_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {file_path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.shape[0] != n_items:
        raise ValueError(f"Unexpected label count in {file_path}.")
    return data


def load_and_prepare_data(
    train_images_path,
    train_labels_path,
    test_images_path,
    test_labels_path,
    train_samples,
    test_samples,
):
    x_train = read_idx_images(train_images_path)[:train_samples].astype(float) / 255.0
    y_train = read_idx_labels(train_labels_path)[:train_samples].astype(int)
    x_test = read_idx_images(test_images_path)[:test_samples].astype(float) / 255.0
    y_test = read_idx_labels(test_labels_path)[:test_samples].astype(int)

    y_train_oh = one_hot(y_train, num_classes=10)
    y_test_oh = one_hot(y_test, num_classes=10)

    return x_train, y_train_oh, x_test, y_test_oh


def run_mnist_experiment(epochs, train_samples, test_samples, batch_size):
    import kagglehub

    dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    train_images, train_labels, test_images, test_labels = find_mnist_idx_files(dataset_path)

    if any(path is None for path in [train_images, train_labels, test_images, test_labels]):
        raise FileNotFoundError(
            "Could not find required MNIST IDX files inside "
            f"{dataset_path} (train-images/labels and t10k-images/labels)."
        )

    x_train, y_train, x_test, y_test = load_and_prepare_data(
        train_images,
        train_labels,
        test_images,
        test_labels,
        train_samples,
        test_samples,
    )

    model = MultiLayerPerceptron(
        x=x_train,
        y=y_train,
        learning_rate=0.001,
        loss_function=CrossEntropy(),
        optimizer=AdamOptimizer(learning_rate=0.001),
    )
    model.add_layer(num_neurons=128, activation_function=ReLUFunction())
    model.add_layer(num_neurons=64, activation_function=ReLUFunction())
    model.add_layer(num_neurons=10, activation_function=SoftmaxFunction())

    history = model.train(
        epochs=epochs,
        verbose=True,
        batch_size=batch_size,
        shuffle=True,
        metrics=[CategoricalAccuracy()],
    )

    eval_result = model.evaluate(x_test, y_test, metrics=[CategoricalAccuracy()])
    sample_idx = 0
    sample_proba = model.predict(x_test[sample_idx : sample_idx + 1])[0]
    sample_pred = int(np.argmax(sample_proba))
    sample_conf = float(sample_proba[sample_pred])
    sample_true = int(np.argmax(y_test[sample_idx]))

    print(f"\nTrain final loss: {history['loss'][-1]:.6f}")
    print(f"Test loss: {eval_result['loss']:.6f}")
    print(f"Test categorical_accuracy: {eval_result['categorical_accuracy']:.6f}")
    print("\nSingle-item prediction (test sample #0):")
    print(f"Predicted class: {sample_pred}")
    print(f"Confidence: {sample_conf:.6f}")
    print(f"True class: {sample_true}")


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST experiment using the MLP package.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mnist_experiment(
        epochs=args.epochs,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
    )
