MECHANISM: Single-coordinate attention LayerNorm bias absorption

HYPOTHESIS: Pruning one `ln1` bias coordinate will reduce parameters from 1,595 to 1,594 while retaining at least 99% accuracy, because its effect can be represented by the downstream QKV and attention-output biases.

INTENDED_EDIT: Add a one-coordinate-pruned LayerNorm and use it before causal self-attention, leaving the successful two-coordinate `ln2` pruning unchanged.

EVIDENCE: Two `ln2` bias coordinates were pruned while achieving 99.97% accuracy; because pruning a third fell to 98.12%, the most informative next step is the smallest equivalent bias-absorption test in the previously unpruned `ln1`.

<<<<<<< SEARCH
class TwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two bias coordinates absorbed by the following linear bias."""
=======
class OnePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with one bias coordinate absorbed by downstream attention biases."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class TwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two bias coordinates absorbed by the following linear bias."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = OnePrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE