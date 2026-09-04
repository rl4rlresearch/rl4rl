MECHANISM: Downstream-absorbed sixth-coordinate LayerNorm scale

HYPOTHESIS: Fixing `ln2` scale coordinate five at one will reduce the model from 1,527 to 1,526 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.

INTENDED_EDIT: Retain learned `ln2` scales only for coordinates zero, three, and six, reconstructing coordinates one, two, four, five, and seven as ones.

EVIDENCE: Fixing coordinate four—the first fully connected `fc1` input coordinate tested—preserved 99.97% accuracy at 1,527 parameters, while failures occurred on coordinates zero and three, which are each omitted by one row. Coordinate five has the same fully connected downstream structure as successful coordinate four.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and four scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (full_weight[:1], full_weight[3:4], full_weight[5:-1])
            ).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:2],
                self.weight.new_ones(1),
                self.weight[2:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
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
>>>>>>> REPLACE