MECHANISM: Fully connected LayerNorm-scale absorption

HYPOTHESIS: Fixing `ln2` scale coordinate four at one will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because all twelve `fc1` rows have trainable coefficients for that coordinate and can absorb its scale.

INTENDED_EDIT: Retain learned `ln2` scales for coordinates zero, three, five, and six, reconstructing coordinates one, two, four, and seven as ones.

EVIDENCE: Fixing coordinates seven, one, and two preserved at least 99.96% accuracy, while failed coordinates zero and three are each structurally omitted by one `fc1` row. Coordinate four is the first untested scale used by every `fc1` row, making downstream absorption least constrained.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and three scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat((full_weight[:1], full_weight[3:-1])).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(2),
                self.weight[1:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
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
>>>>>>> REPLACE