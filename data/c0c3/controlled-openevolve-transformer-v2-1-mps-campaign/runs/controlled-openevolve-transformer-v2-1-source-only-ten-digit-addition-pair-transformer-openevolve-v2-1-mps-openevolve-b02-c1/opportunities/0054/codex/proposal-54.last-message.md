MECHANISM: Single-coordinate final-normalization bias tie

HYPOTHESIS: Tying one final LayerNorm bias coordinate to its common-mode reference will reduce the model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because it preserves six learned feature offsets rather than deleting the entire essential bias pathway.

INTENDED_EDIT: Store six final LayerNorm bias coordinates and pad two reference-valued coordinates; the existing quotient optimizer automatically trains the resulting seven-coordinate virtual bias representation.

EVIDENCE: Removing all seven final LayerNorm bias coordinates collapsed accuracy to 15.64%, while removing the seven-parameter MLP residual bias retained 99.95%; this makes a one-coordinate final-bias ablation the most informative test of whether the essential offset pathway contains any residual redundancy.

<<<<<<< SEARCH
class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm modulo common bias and residual-scale gauges."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 1)),
            self.eps,
        )
=======
class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm with two bias coordinates tied as a reference."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            F.pad(self.bias, (0, 2)),
            self.eps,
        )
>>>>>>> REPLACE