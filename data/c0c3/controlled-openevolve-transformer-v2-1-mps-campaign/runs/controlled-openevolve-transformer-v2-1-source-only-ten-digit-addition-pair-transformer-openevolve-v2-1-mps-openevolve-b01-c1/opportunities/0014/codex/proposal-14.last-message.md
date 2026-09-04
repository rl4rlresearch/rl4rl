MECHANISM: Incremental MLP LayerNorm bias absorption

HYPOTHESIS: Fixing a second `ln2` bias coordinate to zero will reduce parameters from 1,616 to 1,615 while retaining at least 99% accuracy, because both omitted coordinates can be absorbed into the retained `fc1` bias.

INTENDED_EDIT: Retain six of eight second pre-LayerNorm bias coordinates and reconstruct the final two as fixed zeros, leaving all other model capacity and training settings unchanged.

EVIDENCE: Pruning one `ln2` bias coordinate achieved 99.98% accuracy, and the following affine `fc1` bias provides an exact representational replacement; this is the smallest extension of that successful compression.

<<<<<<< SEARCH
class SinglePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with one bias coordinate absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class TwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two bias coordinates absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = SinglePrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE