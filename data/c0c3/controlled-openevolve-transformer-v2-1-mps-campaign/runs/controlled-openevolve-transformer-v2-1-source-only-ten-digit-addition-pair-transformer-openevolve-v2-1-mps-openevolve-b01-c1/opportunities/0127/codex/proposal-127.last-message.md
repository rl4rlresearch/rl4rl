MECHANISM: Adaptive shared LayerNorm scale

HYPOTHESIS: Tying `ln2` scale coordinate six to the mean of the load-bearing scales at coordinates zero and three will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because coordinate six remains trainable instead of being fixed at one.

INTENDED_EDIT: Retain independent `ln2` scales for coordinates zero and three, and reconstruct coordinate six as their arithmetic mean.

EVIDENCE: Fixing coordinate six at one fell to 66.5%, whereas fixing coordinates four and five retained 99.97% and 99.82%; this suggests the final scale needs adaptive training dynamics, which parameter sharing preserves while testing whether it needs an independent degree of freedom.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and five scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (full_weight[:1], full_weight[3:4], full_weight[6:-1])
            ).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(2),
                self.weight[2:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and a shared adaptive final scale."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat((full_weight[:1], full_weight[3:4])).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_final_scale = self.weight.mean().reshape(1)
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(2),
                shared_final_scale,
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE