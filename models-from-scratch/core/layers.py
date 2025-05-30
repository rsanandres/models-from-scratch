import numpy as np
from activations import sigmoid, sigmoid_deriv, relu, relu_deriv

class Dense:
    """Fully connected (dense) layer."""
    def __init__(self, in_features, out_features, activation='sigmoid'):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        self.activation = activation
        self.last_input = None
        self.last_z = None

    def forward(self, x):
        self.last_input = x
        z = x @ self.W + self.b
        self.last_z = z
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        else:
            return z

    def backward(self, grad_output):
        if self.activation == 'sigmoid':
            grad_activation = sigmoid_deriv(self.last_z)
        elif self.activation == 'relu':
            grad_activation = relu_deriv(self.last_z)
        else:
            grad_activation = 1
        grad = grad_output * grad_activation
        grad_W = self.last_input.T @ grad
        grad_b = np.sum(grad, axis=0, keepdims=True)
        grad_input = grad @ self.W.T
        return grad_input, grad_W, grad_b 