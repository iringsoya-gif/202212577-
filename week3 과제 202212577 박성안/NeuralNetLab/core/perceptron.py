import numpy as np


GATE_DATA = {
    'AND': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        'y': np.array([0, 0, 0, 1], dtype=float),
    },
    'OR': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        'y': np.array([0, 1, 1, 1], dtype=float),
    },
    'XOR': {
        'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        'y': np.array([0, 1, 1, 0], dtype=float),
    },
}


class Perceptron:
    def __init__(self, input_size: int = 2, learning_rate: float = 0.1):
        self.input_size = input_size
        self.lr = learning_rate
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.history: list[dict] = []

    def _activation(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def predict(self, inputs) -> int:
        total = np.dot(self.weights, inputs) + self.bias
        return self._activation(total)

    def train_one_epoch(self, X, y) -> int:
        errors = 0
        for xi, yi in zip(X, y):
            pred = self.predict(xi)
            delta = yi - pred
            if delta != 0:
                self.weights += self.lr * delta * xi
                self.bias += self.lr * delta
                errors += 1
        return errors

    def train(self, X, y, epochs: int = 100) -> list:
        self.history = []
        for epoch in range(epochs):
            errors = self.train_one_epoch(X, y)
            self.history.append({
                'epoch': epoch + 1,
                'w1': self.weights[0],
                'w2': self.weights[1],
                'b': self.bias,
                'errors': errors,
            })
            if errors == 0:
                break
        return self.history

    def reset(self):
        self.weights = np.zeros(self.input_size)
        self.bias = 0.0
        self.history = []

    def get_decision_boundary(self):
        return self.weights[0], self.weights[1], self.bias

    def evaluate(self, X, y):
        predictions = np.array([self.predict(xi) for xi in X])
        accuracy = np.mean(predictions == y) * 100
        return predictions, accuracy
