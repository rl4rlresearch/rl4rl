MECHANISM: Redundant value-bias elimination

HYPOTHESIS: Eliminating all seven remaining value-projection bias parameters will reduce the model from 1,625 to 1,618 parameters while retaining at least 99% accuracy, because attention weights sum to one at the configured zero dropout, making the value bias a position-independent output offset already represented by the attention output bias up to the common mode removed by `ln2`.

INTENDED_EDIT: Retain the eight learned query-bias coordinates while fixing every key and value bias coordinate to zero.

EVIDENCE: The seven-dimensional orthonormal attention-output bias achieved 99.89% at 1,625 parameters, showing that the complete output-bias contrast space trains successfully; the remaining value bias only supplies another parameterization of that same downstream offset.

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
    """Combined QKV projection with only a learned query bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:d_model].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.bias, self.bias.new_zeros(2 * self.d_model))
        )
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE