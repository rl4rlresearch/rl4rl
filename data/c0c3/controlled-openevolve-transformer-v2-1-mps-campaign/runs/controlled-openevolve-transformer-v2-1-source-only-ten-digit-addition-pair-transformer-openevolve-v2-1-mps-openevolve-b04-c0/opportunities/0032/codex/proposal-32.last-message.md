MECHANISM: Second pre-MLP scale-to-weight gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 3 to one will reduce the model from 1615 to 1614 parameters while retaining at least 99% accuracy, because its scale is exactly absorbable by column 3 of `fc1.weight`, its bias is already fixed at zero, and coordinate 3 previously tolerated bias gauge fixing.

INTENDED_EDIT: Remove gain coordinate 3 from the learned parameter vector and reconstruct gains 3 and 7 as fixed ones during the forward pass.

EVIDENCE: Fixing gain coordinate 7 achieved 99.97% accuracy at 1615 parameters, establishing this scale-to-weight reduction; coordinate 3 is the strongest next candidate because fixing its bias retained 99.93% at 1616 parameters and its distributed embedding anchor previously achieved 99.98%.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gain 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 1))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        weight = torch.cat((self.weight_rest, self.weight_rest.new_ones(1)))
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with four fixed shifts and gains 3 and 7 absorbed by the MLP."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight_rest = nn.Parameter(torch.ones(d_model - 2))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat(
            (self.weight_rest[:3], ones(1), self.weight_rest[3:], ones(1))
        )
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE