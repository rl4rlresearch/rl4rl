MECHANISM: Shared post-attention LayerNorm/MLP bias contrast

HYPOTHESIS: Reusing `fc1.bias[4]` as the fifth-coordinate `ln2` contrast will reduce the model to 1609 parameters while retaining at least 99% accuracy, because it preserves the adaptive coordinate that rescued the 1610-parameter design.

INTENDED_EDIT: Store three independent `ln2` contrasts and reconstruct the fifth active coordinate from the existing fifth MLP input-bias parameter.

EVIDENCE: Retaining the fifth `ln2` coordinate achieved 99.77% at 1610 parameters, while sharing the required `ln1` contrast with an existing query bias achieved 100% where fixing it to zero achieved only 97.43%; this motivates adaptive sharing rather than deleting the important `ln2` contrast.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat(
            (
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
                self.bias.new_zeros(2),
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
        self.bias = nn.Parameter(torch.zeros(d_model - 5))
        self.eps = 1e-5

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                self.bias,
                self.bias.new_zeros(1),
                shared_bias.reshape(1),
                self.bias.new_zeros(2),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = x + self.mlp(self.ln2(x))
=======
        x = x + self.mlp(self.ln2(x, self.mlp.fc1.bias[4]))
>>>>>>> REPLACE