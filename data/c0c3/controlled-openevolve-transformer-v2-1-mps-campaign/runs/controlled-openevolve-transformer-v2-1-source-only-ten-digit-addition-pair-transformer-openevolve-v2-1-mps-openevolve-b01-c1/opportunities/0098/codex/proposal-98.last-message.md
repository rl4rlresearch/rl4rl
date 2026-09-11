MECHANISM: Final-scale/downstream-column gauge fixing

HYPOTHESIS: Fixing the final `ln2` scale at one will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, because its effect remains representable by the learned final input column of `fc1` and fresh initialization is unchanged.

INTENDED_EDIT: Remove the final trainable scale coordinate from the bias-free MLP LayerNorm and reconstruct it as a fixed one during the forward pass.

EVIDENCE: The 1,535-parameter design achieved 99.87% accuracy with seven `fc1` rows already independent of the final input coordinate, while four attempts to constrain the remaining row 5 failed. This motivates preserving row 5 and instead testing the exact upstream-scale/downstream-column gauge, which affects only the five rows retaining that column.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm whose bias is absorbed completely by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
=======
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and final scale fixed by the downstream column gauge."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE