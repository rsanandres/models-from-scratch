# Models from Scratch

Neural network implementations in pure NumPy — no frameworks, no autograd. Every forward pass, backward pass, and gradient update is written by hand.

## Implementations

### Multi-Layer Perceptron

Configurable MLP with arbitrary hidden layers and per-layer activations. Includes a full training pipeline with hand-derived backpropagation, SGD optimization, and He weight initialization.

Demo: solves XOR classification with a `[2, 2, 1]` architecture.

### Convolutional Neural Network

End-to-end CNN built from custom components:

- `Conv2D` — multi-channel convolution with Xavier initialization
- `MaxPool2D` — max pooling with mask-based gradient routing
- `Flatten` — shape-preserving reshape for the backward pass
- `Dense` — fully connected output layer

Complete backward pass computes gradients through the convolution operation.

### Transformer Self-Attention

Scaled dot-product attention with Q, K, V projection matrices and numerically stable softmax.

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
```

## Project Structure

```
models-from-scratch/
├── core/
│   ├── activations.py      # sigmoid, relu (forward + derivatives)
│   ├── layers.py           # Dense layer with forward/backward
│   ├── losses.py           # MSE loss + derivative
│   └── optimizers.py       # SGD optimizer
└── models/
    ├── mlp.py              # Multi-layer perceptron
    ├── cnn.py              # Convolutional neural network
    └── llm.py              # Self-attention mechanism
```

## Requirements

```
numpy
```

## License

MIT
