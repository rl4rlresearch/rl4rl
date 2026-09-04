MECHANISM: One-coordinate pre-MLP LayerNorm-scale quotient

HYPOTHESIS: Fixing one `ln2` scale coordinate at one will reduce the verified 1,286-parameter model to 1,285 parameters while retaining at least 99% accuracy, because `fc1` can absorb that channel scale and the analogous pre-attention quotient retained accuracy with seven fixed coordinates.

INTENDED_EDIT: Represent `ln2` with seven learned scales and one fixed unit scale, leaving the learned attention, MLP width, initialization, optimizer structure, checkpoints, and decoding interface unchanged.

EVIDENCE: The current 1,286-parameter design achieved 99.93% accuracy, and the verified `ln1` sequence retained at least 99% accuracy while progressively fixing one through seven scale coordinates; this motivates the conservative first-coordinate test at `ln2`.

<<<<<<< SEARCH
class GaugeFixedValueLinear(nn.Linear):
    """Value map modulo its LayerNorm-nullspace row shifts."""
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with one scale absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )


class GaugeFixedValueLinear(nn.Linear):
    """Value map modulo its LayerNorm-nullspace row shifts."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = GaugeFixedMLPScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE