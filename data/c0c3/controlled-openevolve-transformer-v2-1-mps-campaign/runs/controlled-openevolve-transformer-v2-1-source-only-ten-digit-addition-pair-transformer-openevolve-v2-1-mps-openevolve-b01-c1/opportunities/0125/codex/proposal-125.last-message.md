MECHANISM: Fully connected final LayerNorm-scale absorption

HYPOTHESIS: Fixing `ln2` scale coordinate six at one will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.

INTENDED_EDIT: Retain learned `ln2` scales only for coordinates zero and three, reconstructing coordinates one, two, and four through seven as ones.

EVIDENCE: Successively fixing fully connected coordinates four and five preserved 99.97% and 99.82% accuracy; coordinate six has the same full downstream connectivity, unlike failed coordinates zero and three, which are each omitted by an `fc1` row.

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
    """LayerNorm with absorbed bias and six scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat((full_weight[:1], full_weight[3:4])).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:],
                self.weight.new_ones(4),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE