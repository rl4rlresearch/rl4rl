MECHANISM: Third-coordinate LayerNorm-bias quotient

HYPOTHESIS: Fixing `ln1` bias coordinate two at zero will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because it is the only untested coordinate among the four currently trainable biases and prior quotient results demonstrate strong coordinate-specific optimization behavior.

INTENDED_EDIT: Retain trainable `ln1` bias coordinates zero, one, and three, while reconstructing coordinate two and the final four coordinates as zeros without changing fresh zero-bias initialization.

EVIDENCE: Removing currently trainable `ln1` bias coordinates three, zero, and one yielded 11.19%, 56.24%, and 75.50% accuracy respectively, while coordinate-specific changes previously rescued MLP quotients—including row 5 succeeding only on its second coordinate—making the remaining coordinate-two test the most targeted next reduction.

<<<<<<< SEARCH
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with five bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(
            torch.cat((full_bias[:2], full_bias[3:4])).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
                self.bias.new_zeros(4),
            )
        )
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE