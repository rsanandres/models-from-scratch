import numpy as np

# --- Self-Attention Mechanism ---
class SelfAttention:
    """
    Simplified self-attention mechanism (single-head, no masking, no positional encoding).
    Demonstrates the core matrix operations of dot-product attention.
    """
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # Initialize weights for query, key, value
        self.W_q = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_attention = None
        self.last_input = None

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, embed_dim)
        """
        self.last_input = x
        Q = x @ self.W_q  # (batch, seq_len, embed_dim)
        K = x @ self.W_k
        V = x @ self.W_v
        self.last_q, self.last_k, self.last_v = Q, K, V
        # Compute scaled dot-product attention
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.embed_dim)  # (batch, seq_len, seq_len)
        attention = softmax(scores, axis=-1)  # (batch, seq_len, seq_len)
        self.last_attention = attention
        out = attention @ V  # (batch, seq_len, embed_dim)
        return out

    def backward(self, grad_output):
        # For demonstration, we skip full backprop for attention (complex, out of scope for a minimal example)
        # In a real implementation, you would compute gradients for W_q, W_k, W_v and propagate through softmax.
        pass

# --- Utility Functions ---
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# --- Example Usage ---
if __name__ == "__main__":
    # Example: 2 sequences, each of length 4, embedding dimension 8
    np.random.seed(42)
    batch = 2
    seq_len = 4
    embed_dim = 8
    X = np.random.randn(batch, seq_len, embed_dim)
    attn = SelfAttention(embed_dim)
    out = attn.forward(X)
    print("Input shape:", X.shape)
    print("Output shape:", out.shape)
    print("Output (rounded):\n", np.round(out, 2))
    # This demonstrates the core matrix operations of self-attention. 