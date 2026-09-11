MECHANISM: Fifth pre-MLP bias gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm bias coordinate 5 in addition to coordinates 0, 1, 3, and 7 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its shift is absorbable by `fc1.bias` and its distributed embedding anchor previously achieved 99.97%.

INTENDED_EDIT: Remove LayerNorm bias coordinate 5 from the learned parameter vector and reconstruct it as zero during the forward pass.

EVIDENCE: Fixing coordinates 0, 1, 3, and 7 achieved 99.93% at 1616 parameters. Among the remaining coordinates, coordinate 5 has the strongest associated distributed-anchor result—99.97%—while the known-sensitive coordinate 2 remains learned.

<<<<<<< SEARCH
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
=======
class MLPAnchoredLayerNorm(nn.Module):
    """LayerNorm with shifts 0, 1, 3, 5, and 7 absorbed by the MLP bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        bias = torch.cat(
            (
                zeros(2),
                self.bias_rest[:1],
                zeros(1),
                self.bias_rest[1:2],
                zeros(1),
                self.bias_rest[2:],
                zeros(1),
            )
        )
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE