MECHANISM: Complete pre-attention LayerNorm-scale absorption

HYPOTHESIS: Fixing the final adaptive `ln1` scale will reduce the qualified model from 1,278 to 1,277 learned parameters while retaining at least 99% accuracy, because the following value map can absorb every channel scale.

INTENDED_EDIT: Replace the sole learned `ln1` scale with a fixed unit-scale buffer while preserving the learned attention, optimizer, checkpoints, and decoding interface.

EVIDENCE: The current complete `ln2` scale absorption achieved 99.94% accuracy with 1,278 parameters, and the analogous seven-coordinate `ln1` quotient previously achieved 99.96%; this motivates removing its sole remaining adaptive scale.

<<<<<<< SEARCH
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with seven scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(7)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
=======
class GaugeFixedScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with all scales absorbed into the following value map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model))

    def full_weight(self) -> torch.Tensor:
        return self.fixed_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
>>>>>>> REPLACE