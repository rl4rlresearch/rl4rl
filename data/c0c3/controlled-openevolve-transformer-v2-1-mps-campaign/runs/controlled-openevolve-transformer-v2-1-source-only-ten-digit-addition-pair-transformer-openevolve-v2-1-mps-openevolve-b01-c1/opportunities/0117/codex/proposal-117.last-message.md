MECHANISM: Downstream-absorbed second LayerNorm scale

HYPOTHESIS: Fixing `ln2` scale coordinate one at its initialized value of one will reduce the model from 1,530 to 1,529 parameters while retaining at least 99% accuracy, because its effect remains absorbable by the ten trainable `fc1` coefficients that use that coordinate.

INTENDED_EDIT: Retain learned `ln2` scales for coordinates zero and two through six, while reconstructing coordinates one and seven as ones during the forward pass.

EVIDENCE: Fixing coordinate seven achieved 99.96% accuracy at 1,530 parameters; coordinate one is the next-sparsest `fc1` input coordinate because the successful row-five and row-six quotients already omit it.

<<<<<<< SEARCH
class BiasFreeMLPLayerNorm(nn.LayerNorm):
    """LayerNorm with absorbed bias and final scale fixed at one."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
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
>>>>>>> REPLACE