MECHANISM: Distributed-coordinate MLP bias gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm bias coordinate 3 in addition to coordinates 0, 1, and 7 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its shift is exactly absorbable by `fc1.bias` and coordinate 3 previously supported the strongest distributed-anchor result.

INTENDED_EDIT: Remove LayerNorm bias coordinate 3 from the learned parameter vector and reconstruct it as zero during the forward pass.

EVIDENCE: Fixing coordinates 0, 1, and 7 achieved 99.90% at 1617 parameters, while the distributed embedding anchor associated with coordinate 3 achieved 99.98%; unlike the failing coordinate 2, coordinate 3 uses a separate token-row endpoint.

<<<<<<< SEARCH
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with the first two and final shifts absorbed by the MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat((zeros(2), self.bias_rest, zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with shifts 0, 1, 3, and 7 absorbed by the MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat(
            (zeros(2), self.bias_rest[:1], zeros(1), self.bias_rest[1:], zeros(1))
        )
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE