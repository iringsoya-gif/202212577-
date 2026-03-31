import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1.0 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_d(x):
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_d(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def linear(x):
    return x


def linear_d(x):
    return np.ones_like(x)


FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_d),
    'tanh':    (tanh,    tanh_d),
    'relu':    (relu,    relu_d),
    'leaky':   (leaky_relu, leaky_relu_d),
    'linear':  (linear,  linear_d),
}

DISPLAY_FUNCTIONS = {
    'Sigmoid': (sigmoid, sigmoid_d, '1/(1+e⁻ˣ)', '#FF6B6B'),
    'Tanh':    (tanh,    tanh_d,    'tanh(x)',    '#4FC3F7'),
    'ReLU':    (relu,    relu_d,    'max(0,x)',   '#69F0AE'),
    'Leaky':   (leaky_relu, leaky_relu_d, 'max(αx,x)', '#FFD740'),
}
