MECHANISM: Downstream-absorbed third-coordinate LayerNorm scale

HYPOTHESIS: Fixing `ln2` scale coordinate two at one will reduce the model from 1,529 to 1,528 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.

INTENDED_EDIT: Retain learned `ln2` scales for coordinates zero and three through six, reconstructing coordinates one, two, and seven as ones.

EVIDENCE: Fixing `ln2` coordinates seven and one successively preserved 99.96% and 99.99% accuracy; coordinate two is among the next-sparsest `fc1` input coordinates and has also shown favorable optimization behavior in the successful positional-coordinate quotient.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and two scales fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat((full_weight[:1], full_weight[2:-1])).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
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
>>>>>>> REPLACE