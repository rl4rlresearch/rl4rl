MECHANISM: Final query-bias anchoring

HYPOTHESIS: Fixing query-bias coordinate 7 at zero will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because initialization is unchanged and the verified model already succeeds with all value biases fixed and no independently learned key biases.

INTENDED_EDIT: Store seven query-bias coordinates in `KeyAnchoredLinear`, reconstruct the eighth as zero, and preserve the existing key-bias sharing and zero value bias.

EVIDENCE: The 1,576-parameter design achieved 99.92% despite removing all independent key and value biases, while every recent 1,575-parameter attempt constrained LayerNorm, embedding, or residual-output quotient coordinates and failed; testing the remaining attention-bias family is therefore the most informative orthogonal reduction.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-16].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                self.bias[: self.d_model],
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with an anchored query bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-17].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_bias = F.pad(self.bias, (0, 1))
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                query_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                self.bias.new_zeros(3),
                shared_key_bias,
                value_bias,
            )
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE