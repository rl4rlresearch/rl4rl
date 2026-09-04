MECHANISM: Mean-fixed terminal scale gauge

HYPOTHESIS: Constraining the final LayerNorm scale to have mean one will reduce the verified 1,535-parameter model to 1,534 parameters while retaining at least 99% accuracy, because any positive global multiplier of its scale only changes logit temperature and not greedy predictions, while all seven relative feature scales remain learnable.

INTENDED_EDIT: Reparameterize the eight-element terminal normalization scale as an all-ones vector plus seven learned mean-zero coordinates.

EVIDENCE: The current gauge-fixed model reached 99.63% accuracy with 1,535 parameters after removing the final LayerNorm bias; this motivates a minimal follow-up that preserves every relative terminal scale rather than removing the full scale vector.

<<<<<<< SEARCH
class ScaleOnlyLayerNorm(nn.Module):
    """Layer normalization with a learned scale and no learned shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, None)
=======
class ScaleOnlyLayerNorm(nn.Module):
    """Layer normalization with a mean-fixed learned scale and no shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight_offset = nn.Parameter(torch.zeros(dim - 1))
        self.register_buffer("basis", mean_zero_basis(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.basis @ self.weight_offset
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE