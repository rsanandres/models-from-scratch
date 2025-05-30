import numpy as np
from layers import Dense
from optimizers import SGD
from losses import mse_loss, mse_loss_deriv

class MLP:
    """
    Basic Multi-layer Perceptron (MLP) implemented from scratch using NumPy.
    Supports arbitrary number of hidden layers, sigmoid or ReLU activations, and MSE loss.
    
    Mathematical Intuition:
    - Each layer computes a weighted sum of its inputs, adds a bias, and applies a non-linear activation function.
    - The network learns by minimizing a loss function (e.g., Mean Squared Error) using Stochastic Gradient Descent (SGD),
      updating weights and biases via backpropagation (the chain rule of calculus).
    """
    def __init__(self, layer_sizes, activations, lr=0.1):
        """
        Initializes the MLP.
        Args:
            layer_sizes: list of ints, e.g. [2, 4, 1] for 2-input, 1 hidden layer with 4 units, 1 output
            activations: list of str, e.g. ['relu', 'sigmoid']
            lr: learning rate for SGD
        """
        assert len(layer_sizes) - 1 == len(activations)
        self.layers = []
        # Create each layer with the specified activation
        for i in range(len(activations)):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i+1], activation=activations[i]))
        self.optimizer = SGD(lr)

    def forward(self, x):
        """
        Forward pass through all layers.
        Args:
            x: Input data (numpy array)
        Returns:
            Output of the network after passing through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)  # Pass data through each layer
        return x

    def backward(self, loss_grad):
        """
        Backward pass through all layers (backpropagation).
        Args:
            loss_grad: Gradient of the loss with respect to the output
        Returns:
            grads_W: List of gradients for weights
            grads_b: List of gradients for biases
        """
        grads_W = []
        grads_b = []
        grad = loss_grad
        # Propagate gradients backward through each layer
        for layer in reversed(self.layers):
            grad, grad_W, grad_b = layer.backward(grad)
            grads_W.insert(0, grad_W)  # Insert at the beginning to maintain order
            grads_b.insert(0, grad_b)
        return grads_W, grads_b

    def train(self, X, y, epochs=1000, verbose=100):
        """
        Training loop for the MLP using SGD.
        Args:
            X: Input data
            y: Target labels
            epochs: Number of training epochs
            verbose: Print loss every 'verbose' epochs
        """
        for epoch in range(1, epochs+1):
            # Forward pass: compute predictions
            y_pred = self.forward(X)
            # Compute loss
            loss = mse_loss(y, y_pred)
            # Backward pass: compute gradients
            loss_grad = mse_loss_deriv(y, y_pred)
            grads_W, grads_b = self.backward(loss_grad)
            # Update weights and biases using SGD
            params = [layer.W for layer in self.layers] + [layer.b for layer in self.layers]
            grads = grads_W + grads_b
            self.optimizer.step(params, grads)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.
        Args:
            X: Input data
        Returns:
            Network output (predictions)
        """
        y_pred = self.forward(X)
        return y_pred

# Example Usage: XOR Problem
if __name__ == "__main__":
    # XOR dataset: input and expected output
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Define MLP:
    # - 2 input features
    # - 1 hidden layer with 2 units (ReLU activation)
    # - 1 output unit (Sigmoid activation)
    # This architecture is sufficient to solve the XOR problem, which is not linearly separable.
    mlp = MLP(layer_sizes=[2, 2, 1], activations=['relu', 'sigmoid'], lr=0.1)
    # Train the network on the XOR data
    mlp.train(X, y, epochs=5000, verbose=500)

    # Make predictions after training
    preds = mlp.predict(X)
    print("Predictions after training:")
    print(np.round(preds, 3))
    # The output should be close to [0, 1, 1, 0] for the four XOR inputs 