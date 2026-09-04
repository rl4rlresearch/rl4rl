MECHANISM: Cross-module value/output-bias tying

HYPOTHESIS: Reusing an attention output-projection bias coordinate as the second-to-last value-bias coordinate will reduce the model to 1,631 parameters while retaining at least 99% accuracy, because value bias is converted into a constant residual offset that the output-projection bias already parameterizes.

INTENDED_EDIT: Store six independent value-bias coordinates, retain the successful first-to-last value-bias tie, and reconstruct the seventh coordinate from the learned attention projection bias.

EVIDENCE: The first dynamic value-bias tie achieved 99.84% accuracy at 1,632 parameters, while tying a second pair within the value bias reached only 95.37%; cross-module tying tests whether preserving a distinct learned signal from the projection-bias pathway avoids that optimization loss.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-9].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (self.bias[self.d_model :], self.bias[self.d_model : self.d_model + 1])
        )
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with cross-module tied bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-10].clone())

    def forward(
        self, x: torch.Tensor, shared_value_bias: torch.Tensor
    ) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                shared_value_bias,
                self.bias[self.d_model : self.d_model + 1],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
=======
        qkv = self.qkv(x, self.proj.bias[:1])
>>>>>>> REPLACE