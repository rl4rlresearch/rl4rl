MECHANISM: Pre-MLP LayerNorm scale/input-weight gauge

HYPOTHESIS: Fixing the eighth `ln2` scale at its initial value will reduce the model from 1619 to 1618 parameters while retaining at least 99% accuracy, because its effect can be absorbed exactly by the corresponding `fc1` input-weight column and bias.

INTENDED_EDIT: Store seven learned scales in `ReducedBiasLayerNorm` and reconstruct the eighth as one during the forward pass.

EVIDENCE: Removing a second value bias and centering the fourth `fc2` column collapsed to 53.27% and 72.40%, respectively, motivating a distinct pre-GELU gauge whose eliminated scale is redundant with the following unconstrained linear layer.

<<<<<<< SEARCH
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE