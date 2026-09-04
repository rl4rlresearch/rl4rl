MECHANISM: Complete pre-MLP LayerNorm-scale absorption

HYPOTHESIS: Fixing the final adaptive `ln2` scale will reduce the qualified 1,279-parameter model to 1,278 parameters while retaining at least 99% accuracy, because `fc1` can absorb every channel scale.

INTENDED_EDIT: Represent `ln2` with a fixed unit-scale buffer and no learned scale parameters, leaving the learned attention, MLP, optimizer, checkpoints, and decoding interface unchanged.

EVIDENCE: The seven-coordinate `ln2` quotient achieved 99.97% accuracy with 1,279 parameters, and every preceding incremental quotient exceeded 99%, directly motivating removal of the sole remaining adaptive scale.

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with one scale absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight.new_ones(1)))
=======
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
    """LayerNorm with all scales absorbed into the following MLP map."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = None
        self.bias = None
        self.register_buffer("fixed_weight", torch.ones(d_model))

    def full_weight(self) -> torch.Tensor:
        return self.fixed_weight
>>>>>>> REPLACE