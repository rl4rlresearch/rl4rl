MECHANISM: Anchored terminal-scale gauge

HYPOTHESIS: Fixing one final LayerNorm scale coordinate at 1 while learning the other seven will produce a 1,534-parameter model with at least 99% accuracy, because it removes the prediction-invariant global-scale degree of freedom without imposing the failed mean-one constraint that prevented the learned scale average from moving.

INTENDED_EDIT: Replace the affine final LayerNorm with a bias-free normalization whose eight-element scale consists of seven learned coordinates and one fixed anchor.

EVIDENCE: Bias-free terminal normalization achieved 99.63% with 1,535 parameters, while fixing the scale mean at one narrowly failed at 98.87%; anchoring one coordinate tests a less restrictive seven-parameter scale chart that retains variation in the overall scale average.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with seven learned scales and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, (x.size(-1),), weight, None)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE