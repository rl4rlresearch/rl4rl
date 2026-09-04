MECHANISM: Downstream-bias-absorbed LayerNorm anchoring

HYPOTHESIS: Fixing one `ln2` bias coordinate at zero will reduce the model from 1,627 to 1,626 parameters while retaining at least 99% accuracy, because the following biased MLP projection can absorb that LayerNorm offset and zero initialization preserves the original initial function.

INTENDED_EDIT: Add a bias-only anchored LayerNorm and use it for `ln2`, leaving its scale unconstrained and all existing successful reductions unchanged.

EVIDENCE: A single `ln1` bias anchor previously achieved 99.95% accuracy, while tying an additional coordinate within that same bias fell to 97.52%; applying one independent anchor to `ln2` tests the successful single-anchor mechanism without adding another within-vector tie.

<<<<<<< SEARCH
class Block(nn.Module):
=======
class BiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with one additive coordinate fixed at zero."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, (x.size(-1),), self.weight, F.pad(self.bias, (0, 1)), 1e-5
        )


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = BiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE