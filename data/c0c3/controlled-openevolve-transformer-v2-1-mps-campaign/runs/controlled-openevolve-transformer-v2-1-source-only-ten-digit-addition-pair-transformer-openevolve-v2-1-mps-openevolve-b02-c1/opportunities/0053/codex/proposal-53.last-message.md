MECHANISM: Bias-free final normalization

HYPOTHESIS: Removing the seven learned final LayerNorm bias coordinates will reduce the model from 1,046 to 1,039 parameters while retaining at least 99% accuracy, because the successful residual-bias ablation indicates constant feature offsets are dispensable, while the failed MLP input-bias ablation specifically preserves evidence that offsets before the nonlinear bottleneck are essential.

INTENDED_EDIT: Make the final LayerNorm bias-free and remove its deleted bias from the quotient optimizer.

EVIDENCE: Removing the seven-parameter MLP residual bias retained 99.95% accuracy, whereas removing the scalar MLP input bias collapsed accuracy to 37.81%; this motivates testing another post-computation feature offset without disturbing the load-bearing nonlinear bias.

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
    """Bias-free final LayerNorm modulo the residual-scale gauge."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            F.pad(self.weight, (0, 1), value=1.0),
            None,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
        ] + [
>>>>>>> REPLACE