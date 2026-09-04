MECHANISM: MLP LayerNorm scale–linear weight quotient

HYPOTHESIS: Fixing one `ln2` scale coordinate at its initial value of one will reduce parameters from 1,615 to 1,614 while retaining at least 99% accuracy, because the following `fc1` weight column can learn the omitted scale exactly.

INTENDED_EDIT: Retain seven learned `ln2` scale coordinates and six learned bias coordinates, reconstructing the final scale as one and the final two biases as zero.

EVIDENCE: Removing two `ln2` bias coordinates retained 99.97% accuracy, while removing a third bias fell to 98.12%; testing the independent scale–weight redundancy is therefore more informative than repeating the failed bias-pruning direction.

<<<<<<< SEARCH
class TwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class PrunedMLPLayerNorm(nn.LayerNorm):
    """LayerNorm affine parameters modulo redundancies with the following linear."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = PrunedMLPLayerNorm(cfg.d_model)
>>>>>>> REPLACE