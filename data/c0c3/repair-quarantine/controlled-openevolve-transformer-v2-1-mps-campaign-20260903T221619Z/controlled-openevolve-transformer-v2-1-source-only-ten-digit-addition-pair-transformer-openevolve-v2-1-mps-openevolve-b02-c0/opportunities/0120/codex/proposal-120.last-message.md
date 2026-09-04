MECHANISM: Cross-head query-bias coordinate tying

HYPOTHESIS: Tying query-bias coordinate 7 to the homologous coordinate 3 of the other attention head will reduce the model to 1,575 parameters while retaining at least 99% accuracy, because it restores an adaptable offset absent from the nearly successful zero-anchor design.

INTENDED_EDIT: Store seven query-bias parameters and reconstruct coordinate 7 from coordinate 3, preserving initialization and all existing key/value-bias constraints.

EVIDENCE: Fixing query-bias coordinate 7 at zero reached 98.35%, the closest 1,575-parameter result; this motivates retaining that coordinate’s adaptability through the structurally corresponding coordinate of the other four-dimensional head.

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
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with a cross-head query-bias tie."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-17].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_bias = torch.cat((self.bias, self.bias[3:4]))
        shared_key_bias = self.bias[:1]
        value_bias = self.bias.new_zeros(self.d_model)
        bias = torch.cat(
            (
                query_bias,
>>>>>>> REPLACE