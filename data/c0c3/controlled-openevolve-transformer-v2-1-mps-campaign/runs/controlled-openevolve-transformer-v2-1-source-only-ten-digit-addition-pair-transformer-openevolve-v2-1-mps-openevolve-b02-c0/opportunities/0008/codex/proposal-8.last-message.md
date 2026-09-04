MECHANISM: Single-coordinate softmax-invariant key-bias anchoring

HYPOTHESIS: Fixing one key-bias coordinate at zero will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because a position-independent key offset only adds a common shift to each query’s attention logits.

INTENDED_EDIT: Replace the combined QKV linear layer with an initialization-preserving subclass that learns 23 bias coordinates and inserts one fixed-zero coordinate in the key-bias segment.

EVIDENCE: The one-coordinate LayerNorm anchor achieved 99.95% at 1,643 parameters, while removing all key-bias coordinates collapsed accuracy; an incremental one-coordinate key anchor tests the exact attention redundancy with the smallest possible optimization perturbation.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with one fixed-zero key-bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias[: self.d_model], self.bias.new_zeros(1), self.bias[self.d_model :])
        )
        return F.linear(x, self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = KeyAnchoredLinear(d_model)
>>>>>>> REPLACE