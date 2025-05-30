import numpy as np
from core.activations import relu, relu_deriv, sigmoid
from core.optimizers import SGD
from core.losses import mse_loss, mse_loss_deriv

class Conv2D:
    """
    Basic 2D Convolutional Layer (single channel, no padding, stride=1).
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Xavier initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros((out_channels, 1))
        self.last_input = None

    def forward(self, x):
        """
        x: (batch, in_channels, height, width)
        Returns: (batch, out_channels, out_height, out_width)
        """
        self.last_input = x
        batch, in_c, h, w = x.shape
        k = self.kernel_size
        out_h = h - k + 1
        out_w = w - k + 1
        out = np.zeros((batch, self.out_channels, out_h, out_w))
        for b in range(batch):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            out[b, oc, i, j] += np.sum(
                                x[b, ic, i:i+k, j:j+k] * self.W[oc, ic]
                            )
                out[b, oc, :, :] += self.b[oc]
        return out

    def backward(self, grad_output):
        x = self.last_input
        batch, in_c, h, w = x.shape
        k = self.kernel_size
        out_h = h - k + 1
        out_w = w - k + 1
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_input = np.zeros_like(x)
        for b in range(batch):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            grad_W[oc, ic] += grad_output[b, oc, i, j] * x[b, ic, i:i+k, j:j+k]
                            grad_input[b, ic, i:i+k, j:j+k] += grad_output[b, oc, i, j] * self.W[oc, ic]
                grad_b[oc] += np.sum(grad_output[b, oc])
        return grad_input, grad_W, grad_b

class MaxPool2D:
    """
    Basic 2D Max Pooling Layer (no overlap, stride=pool_size).
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.last_input = None
        self.last_mask = None

    def forward(self, x):
        """
        x: (batch, channels, height, width)
        Returns: (batch, channels, out_h, out_w)
        """
        self.last_input = x
        batch, c, h, w = x.shape
        p = self.pool_size
        out_h = h // p
        out_w = w // p
        out = np.zeros((batch, c, out_h, out_w))
        self.last_mask = np.zeros_like(x)
        for b in range(batch):
            for ch in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        window = x[b, ch, i*p:(i+1)*p, j*p:(j+1)*p]
                        max_val = np.max(window)
                        out[b, ch, i, j] = max_val
                        # Mask for backprop
                        mask = (window == max_val)
                        self.last_mask[b, ch, i*p:(i+1)*p, j*p:(j+1)*p] = mask
        return out

    def backward(self, grad_output):
        x = self.last_input
        batch, c, h, w = x.shape
        p = self.pool_size
        out_h = h // p
        out_w = w // p
        grad_input = np.zeros_like(x)
        for b in range(batch):
            for ch in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        mask = self.last_mask[b, ch, i*p:(i+1)*p, j*p:(j+1)*p]
                        grad_input[b, ch, i*p:(i+1)*p, j*p:(j+1)*p] += grad_output[b, ch, i, j] * mask
        return grad_input

class Flatten:
    """
    Flattens the input for fully connected layers.
    """
    def __init__(self):
        self.last_shape = None
    def forward(self, x):
        self.last_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_output):
        return grad_output.reshape(self.last_shape)

class Dense:
    """
    Simple fully connected layer for CNN output.
    """
    def __init__(self, in_features, out_features, activation='relu'):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((1, out_features))
        self.activation = activation
        self.last_input = None
        self.last_z = None
    def forward(self, x):
        self.last_input = x
        z = x @ self.W + self.b
        self.last_z = z
        if self.activation == 'relu':
            return relu(z)
        elif self.activation == 'sigmoid':
            return sigmoid(z)
        else:
            return z
    def backward(self, grad_output):
        if self.activation == 'relu':
            grad_activation = relu_deriv(self.last_z)
        elif self.activation == 'sigmoid':
            grad_activation = sigmoid(z)
        else:
            grad_activation = 1
        grad = grad_output * grad_activation
        grad_W = self.last_input.T @ grad
        grad_b = np.sum(grad, axis=0, keepdims=True)
        grad_input = grad @ self.W.T
        return grad_input, grad_W, grad_b

class SimpleCNN:
    """
    A simple CNN: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Sigmoid
    """
    def __init__(self, in_channels, num_classes, lr=0.1):
        self.conv = Conv2D(in_channels, 2, kernel_size=3)
        self.relu = relu
        self.pool = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc = Dense(2*3*3, num_classes, activation='sigmoid')
        self.optimizer = SGD(lr)
    def forward(self, x):
        x = self.conv.forward(x)
        x = self.relu(x)
        x = self.pool.forward(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return x
    def backward(self, grad_output):
        grad, grad_W_fc, grad_b_fc = self.fc.backward(grad_output)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = relu_deriv(self.conv.last_input) * grad
        grad_input, grad_W_conv, grad_b_conv = self.conv.backward(grad)
        return [grad_W_conv, grad_W_fc], [grad_b_conv, grad_b_fc]
    def train(self, X, y, epochs=100, verbose=10):
        for epoch in range(1, epochs+1):
            y_pred = self.forward(X)
            loss = mse_loss(y, y_pred)
            loss_grad = mse_loss_deriv(y, y_pred)
            grads_W, grads_b = self.backward(loss_grad)
            params = [self.conv.W, self.fc.W, self.conv.b, self.fc.b]
            grads = grads_W + grads_b
            self.optimizer.step(params, grads)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    def predict(self, X):
        return self.forward(X)

# Example Usage: Simple synthetic data (e.g., 2x2 squares in 5x5 images)
if __name__ == "__main__":
    # Create 4 simple 5x5 images, 1 channel
    X = np.zeros((4, 1, 5, 5))
    y = np.array([[1], [0], [0], [1]])
    # Draw a 2x2 square in top-left for class 1, bottom-right for class 0
    X[0, 0, 0:2, 0:2] = 1  # class 1
    X[1, 0, 3:5, 3:5] = 1  # class 0
    X[2, 0, 0:2, 3:5] = 1  # class 0
    X[3, 0, 3:5, 0:2] = 1  # class 1
    cnn = SimpleCNN(in_channels=1, num_classes=1, lr=0.1)
    cnn.train(X, y, epochs=100, verbose=20)
    preds = cnn.predict(X)
    print("Predictions after training:")
    print(np.round(preds, 2)) 