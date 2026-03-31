import numpy as np
from .activation import FUNCTIONS


class NeuralNet:
    """범용 n층 신경망 — Tab3(순전파), Tab5(보편근사) 공용"""

    def __init__(self, layers: list[int], activations: list[str], lr: float = 0.01):
        assert len(activations) == len(layers) - 1
        self.layers = layers
        self.activations = activations
        self.lr = lr
        self.loss_history: list[float] = []
        self._init_params()

    def _init_params(self):
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            w = np.random.randn(fan_in, self.layers[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        # 순전파 캐시
        self._zs: list[np.ndarray] = []
        self._as: list[np.ndarray] = []

    def forward(self, X) -> np.ndarray:
        self._zs = []
        self._as = [np.atleast_2d(X)]
        a = np.atleast_2d(X)
        for w, b, act_name in zip(self.weights, self.biases, self.activations):
            z = a @ w + b
            fn, _ = FUNCTIONS[act_name]
            a = fn(z)
            self._zs.append(z)
            self._as.append(a)
        return a

    def backward(self, X, y):
        m = np.atleast_2d(X).shape[0]
        y = np.atleast_2d(y)
        deltas = [None] * len(self.weights)
        # 출력층 delta
        _, fn_d = FUNCTIONS[self.activations[-1]]
        deltas[-1] = (self._as[-1] - y) * fn_d(self._zs[-1])
        # 역방향 전파
        for i in range(len(self.weights) - 2, -1, -1):
            _, fn_d = FUNCTIONS[self.activations[i]]
            deltas[i] = (deltas[i + 1] @ self.weights[i + 1].T) * fn_d(self._zs[i])
        # 가중치 업데이트
        for i in range(len(self.weights)):
            dw = self._as[i].T @ deltas[i] / m
            db = deltas[i].mean(axis=0, keepdims=True)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def train_step(self, X, y) -> float:
        y_pred = self.forward(X)
        loss = float(np.mean((y_pred - np.atleast_2d(y)) ** 2))
        self.backward(X, y)
        self.loss_history.append(loss)
        return loss

    def train(self, X, y, epochs: int = 1000, callback=None):
        for epoch in range(epochs):
            loss = self.train_step(X, y)
            if callback and epoch % 100 == 0:
                callback(epoch, loss)

    def predict(self, X) -> np.ndarray:
        return self.forward(X)

    def get_layer_values(self) -> dict:
        """Tab3 시각화용 — forward() 호출 후 사용"""
        result = {}
        for i, (z, a) in enumerate(zip(self._zs, self._as[1:])):
            result[f'layer{i+1}'] = {
                'z': z.flatten().tolist(),
                'a': a.flatten().tolist(),
                'act': self.activations[i],
            }
        result['input'] = self._as[0].flatten().tolist()
        return result

    def reset(self):
        self.loss_history = []
        self._init_params()
