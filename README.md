# Multi Layer Perceptron

Implementacao de uma MLP em Python com NumPy, com foco didatico e API simples para treino, avaliacao e predicao.

## Recursos

- Camadas densas com funcoes de ativacao configuraveis
- Otimizadores SGD, Momentum e Adam
- Funcoes de perda para regressao e classificacao
- Suporte a mini-batch, shuffle e metricas no treino
- Testes unitarios cobrindo contrato de APIs e fluxo de treinamento

## Instalacao

Instalacao via GitHub (para usar em outro projeto):

```bash
pip install "git+https://github.com/KWSantos/Multi_Layer_Perceptron.git"
```

Instalacao local para desenvolvimento:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para rodar o exemplo de MNIST:

```bash
pip install -r requirements-examples.txt
```

## Uso rapido

```python
import numpy as np
from mlp import LinearFunction, MeanSquareError, MultiLayerPerceptron

x = np.random.randn(50, 2)
y = 0.7 * x[:, [0]] - 0.2 * x[:, [1]] + 0.1

model = MultiLayerPerceptron(
    x=x,
    y=y,
    learning_rate=0.1,
    loss_function=MeanSquareError(),
)
model.add_layer(num_neurons=1, activation_function=LinearFunction())
model.train(epochs=200, verbose=False)

pred = model.predict(x[:3])
print(pred)
```

## Rodando os testes

```bash
python -m unittest discover -s tests -v
```

## Exemplo completo (MNIST)

```bash
python examples/mnist_experiment.py --epochs 10 --train-samples 3000 --test-samples 1000 --batch-size 64
```

## Estrutura do projeto

```text
mlp/
  activations.py
  losses.py
  metrics.py
  optimizers.py
  core/
  models/
tests/
examples/
```
