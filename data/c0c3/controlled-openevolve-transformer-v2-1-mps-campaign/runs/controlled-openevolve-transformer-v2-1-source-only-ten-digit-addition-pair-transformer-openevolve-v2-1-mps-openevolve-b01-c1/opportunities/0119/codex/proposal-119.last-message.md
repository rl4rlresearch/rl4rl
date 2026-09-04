MECHANISM: Downstream-absorbed fourth-coordinate LayerNorm scale

HYPOTHESIS: Fixing `ln2` scale coordinate three at one will reduce the model from 1,528 to 1,527 parameters while retaining at least 99% accuracy, because its effect can be absorbed by the eleven trainable `fc1` coefficients using that coordinate.

INTENDED_EDIT: Retain learned `ln2` scales for coordinates zero and four through six, reconstructing coordinates one, two, three, and seven as ones.

EVIDENCE: Fixing `ln2` coordinates seven, one, and two successively preserved 99.96%, 99.99%, and 100% accuracy; coordinate three is one of the next-sparsest `fc1` inputs and also exhibited favorable optimization in the successful positional-coordinate quotient.

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
            torch.cat((full_weight[:1], full_weight[4:-1])).clone()
        )
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(3),
                self.weight[1:],
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
>>>>>>> REPLACE