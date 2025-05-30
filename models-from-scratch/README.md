# models-from-scratch

A simple implementation of a Multi-layer Perceptron (MLP) from scratch using NumPy, with support for custom layers, activations, optimizers, and loss functions.

## Features
- Fully connected (Dense) layers
- ReLU and Sigmoid activations
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) optimizer
- Example: Solves the XOR problem

## File Structure
- `mlp.py`: Main MLP class and example usage
- `layers.py`: Dense layer and activation functions
- `activations.py`: Activation functions (ReLU, Sigmoid, etc.)
- `losses.py`: Loss functions and their derivatives
- `optimizers.py`: Optimizer implementations (SGD)

## Usage
Run the example XOR problem:

```bash
python mlp.py
```

## Requirements
- Python 3.7+
- numpy

Install dependencies:
```bash
pip install numpy
```

## License
MIT 