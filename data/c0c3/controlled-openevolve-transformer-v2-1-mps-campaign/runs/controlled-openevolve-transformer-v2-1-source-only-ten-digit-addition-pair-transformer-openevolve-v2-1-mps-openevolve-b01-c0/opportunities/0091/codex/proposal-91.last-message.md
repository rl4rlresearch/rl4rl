MECHANISM: Pre-MLP LayerNorm final-gain gauge fixing

HYPOTHESIS: Fixing the final `ln2` gain to one will retain at least 99% accuracy with 1596 parameters, because its bias coordinate is identically zero and any learned scaling can be absorbed by the unconstrained final input column of `fc1`.

INTENDED_EDIT: Store seven learned gains in `ReducedBiasLayerNorm` and reconstruct the eighth as one before normalization.

EVIDENCE: The 1597-parameter design achieved 99.98%, while reductions targeting additional attention columns and positional coordinates failed; this tests an independent, initialization-preserving scale gauge at the unconstrained MLP interface.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-5

    def forward(
        self,
        x: torch.Tensor,
        third_bias: torch.Tensor,
        fifth_bias: torch.Tensor,
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                third_bias.new_zeros(2),
                third_bias.reshape(1),
                third_bias.new_zeros(1),
                fifth_bias.reshape(1),
                third_bias.new_zeros(2),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, third_bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.eps = 1e-5

    def forward(
        self,
        x: torch.Tensor,
        third_bias: torch.Tensor,
        fifth_bias: torch.Tensor,
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                third_bias.new_zeros(2),
                third_bias.reshape(1),
                third_bias.new_zeros(1),
                fifth_bias.reshape(1),
                third_bias.new_zeros(2),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, third_bias.new_zeros(1)))
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE