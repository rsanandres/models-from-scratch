import numpy as np

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss."""
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_deriv(y_true, y_pred):
    """Derivative of MSE loss with respect to predictions."""
    return 2 * (y_pred - y_true) / y_true.size 