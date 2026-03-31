import numpy as np
from .activation import sigmoid, sigmoid_d


XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_y = np.array([[0], [1], [1], [0]], dtype=float)


class SimpleMLP:
    """Tab4 전용 XOR 학습 2층 MLP — hidden_size 동적 변경 지원"""

    def __init__(self, hidden_size: int = 4, lr: float = 0.1):
        self.lr = lr
        self.reset(hidden_size)

    def reset(self, hidden_size: int = 4):
        self.hidden_size = hidden_size
        # Xavier 초기화
        self.W1 = np.random.randn(2, hidden_size) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        self.loss_history: list[float] = []
        # 역전파용 캐시
        self._z1 = None
        self._a1 = None
        self._z2 = None
        self._a2 = None

    def forward(self, X) -> np.ndarray:
        self._z1 = X @ self.W1 + self.b1
        self._a1 = sigmoid(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = sigmoid(self._z2)
        return self._a2

    def backward(self, X, y):
        m = X.shape[0]
        delta2 = (self._a2 - y) * sigmoid_d(self._z2)
        dW2 = self._a1.T @ delta2 / m
        db2 = delta2.mean(axis=0, keepdims=True)
        delta1 = (delta2 @ self.W2.T) * sigmoid_d(self._z1)
        dW1 = X.T @ delta1 / m
        db1 = delta1.mean(axis=0, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_step(self) -> float:
        """1 epoch, XOR 데이터 고정"""
        y_pred = self.forward(XOR_X)
        loss = float(np.mean((y_pred - XOR_y) ** 2))
        self.backward(XOR_X, XOR_y)
        self.loss_history.append(loss)
        return loss

    def predict_xor(self):
        return (self.forward(XOR_X) > 0.5).astype(int)

    def get_decision_boundary_mesh(self, res: int = 200):
        xx, yy = np.meshgrid(
            np.linspace(-0.5, 1.5, res),
            np.linspace(-0.5, 1.5, res),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        z1 = grid @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        Z = sigmoid(z2).reshape(xx.shape)
        return xx, yy, Z

    def get_hidden_activations(self, X=None) -> np.ndarray:
        if X is None:
            X = XOR_X
        z1 = X @ self.W1 + self.b1
        return sigmoid(z1)

    def get_backprop_info(self) -> dict:
        """역전파 수식 표시용 현재 값 반환"""
        if self._a2 is None:
            return {}
        delta2 = (self._a2 - XOR_y) * sigmoid_d(self._z2)
        dW2 = self._a1.T @ delta2 / 4
        delta1 = (delta2 @ self.W2.T) * sigmoid_d(self._z1)
        dW1 = XOR_X.T @ delta1 / 4
        return {
            'delta2': delta2,
            'dW2': dW2,
            'delta1': delta1,
            'dW1': dW1,
        }
