# Copilot Instructions for this Repository

## Build, test, and lint commands

- Full test suite:
  - `python -m unittest discover -s tests -v`
- Run a single test method:
  - `python -m unittest tests.test_mlp.MLPTests.test_training_reduces_loss_in_simple_regression -v`
- Lint/build:
  - No dedicated lint or build command is currently configured in the repository.

## High-level architecture

- The project is a lightweight MLP framework centered on the `mlp/` package:
  - `mlp/models/mlp.py`: orchestration (`add_layer`, `forward_pass`, `backward_pass`, `train`, `predict`)
  - `mlp/core/layer.py`: per-layer forward/backward logic and parameter updates
  - `mlp/core/neuron.py`: neuron-level parameters (`weights`, `bias`) and linear combination
  - `mlp/activations.py`: activation interfaces and implementations
  - `mlp/losses.py`: loss interfaces and implementations
- `mlp/__init__.py` is the public API surface and re-exports the main classes.
- Current validation lives in `tests/test_mlp.py` and covers gradient-shape behavior plus end-to-end loss reduction in a simple regression scenario.

## Key conventions in this codebase

- **Tensor shape convention:** batch-first is used throughout (`(batch_size, features)`), including layer activations and deltas.
- **Loss function contract:** all losses follow `run(y_true, y_pred)` and `derivative(y_true, y_pred)`.
- **Backprop update order:** in `Layer.backward`, `delta_prev` is computed with pre-update weights (`old_weights`) before parameter updates.
- **Softmax + CrossEntropy path:** `MultiLayerPerceptron.backward_pass` has a dedicated branch that applies the simplified gradient `(y_pred - y_true) / batch_size` and skips extra activation derivative on the final layer.
- **Input normalization:** model inputs are normalized to 2D via `_ensure_2d`; callers should expect 2D behavior in training/inference paths.
- **Deterministic tests:** tests set `np.random.seed(42)` in `setUp`, so new tests should follow this pattern when they depend on initialization randomness.
