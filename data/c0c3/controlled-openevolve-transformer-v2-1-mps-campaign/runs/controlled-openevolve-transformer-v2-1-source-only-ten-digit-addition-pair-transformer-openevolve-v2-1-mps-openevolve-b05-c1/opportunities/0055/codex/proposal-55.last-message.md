MECHANISM: Fixed-coordinate final LayerNorm bias

HYPOTHESIS: Fixing one zero-initialized final LayerNorm bias coordinate will reduce the verified model from 1,315 to 1,314 parameters while retaining at least 99% accuracy within 50,000 steps.

INTENDED_EDIT: Replace the final LayerNorm with an equivalent-at-initialization implementation that learns seven bias coordinates and pads the eighth with zero, while retaining all eight scale parameters.

EVIDENCE: The 1,315-parameter design achieved 100% accuracy, whereas sharing a third MLP bias coordinate fell to 75.02%; testing an output-only, initially unchanged restriction avoids further coupling in the sensitive MLP and leaves seven learned final-output bias directions.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x[..., :-1])


class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one zero-fixed output-bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 1)),
        )


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE