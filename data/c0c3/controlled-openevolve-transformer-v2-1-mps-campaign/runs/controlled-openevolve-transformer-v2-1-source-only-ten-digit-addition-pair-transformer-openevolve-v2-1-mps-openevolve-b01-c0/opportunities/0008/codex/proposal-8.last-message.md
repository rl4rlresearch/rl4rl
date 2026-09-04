MECHANISM: Absorbable attention-input normalization scale

HYPOTHESIS: Fixing one `ln1` gain coordinate at its initialized value of 1 will reduce the model from 1636 to 1635 learned parameters while retaining at least 99% accuracy, because that coordinate’s scale is exactly absorbable by the corresponding QKV weight column and the model begins with identical activations and RNG state.

INTENDED_EDIT: Replace the first block LayerNorm with an equivalent implementation whose final gain coordinate is fixed at 1 while its other gains and all biases remain learned.

EVIDENCE: Initialization-preserving removal of all eight softmax-redundant key biases achieved 99.98%, whereas changing value or projection bias pathways failed; this motivates a minimal reduction in a different exactly redundant parameterization while preserving initial computation.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class ReducedScaleLayerNorm(nn.Module):
    """LayerNorm with one absorbable gain coordinate fixed at its initial value."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, (self.d_model,), weight, self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = ReducedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE