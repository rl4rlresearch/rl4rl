MECHANISM: Coordinate-aligned normalization/query-bias sharing

HYPOTHESIS: Reusing `attn.qv_bias[4]` for the fifth-coordinate `ln2` contrast will reduce the model to 1606 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove the standalone `ln2` bias parameter and source its third- and fifth-coordinate contrasts from the corresponding learned query-bias coordinates.

EVIDENCE: Sharing the third-coordinate `ln2` contrast with `qv_bias[2]` retained both required contrasts and achieved 99.36% at 1607 parameters; applying the same successful coordinate-aligned sharing mechanism to the remaining fifth contrast is the most direct one-parameter reduction.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
        self.eps = 1e-5

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                shared_bias.new_zeros(2),
                shared_bias.reshape(1),
                shared_bias.new_zeros(1),
                self.bias,
                shared_bias.new_zeros(2),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.mlp(self.ln2(x, self.attn.qv_bias[2]))
=======
        x = x + self.mlp(
            self.ln2(x, self.attn.qv_bias[2], self.attn.qv_bias[4])
        )
>>>>>>> REPLACE