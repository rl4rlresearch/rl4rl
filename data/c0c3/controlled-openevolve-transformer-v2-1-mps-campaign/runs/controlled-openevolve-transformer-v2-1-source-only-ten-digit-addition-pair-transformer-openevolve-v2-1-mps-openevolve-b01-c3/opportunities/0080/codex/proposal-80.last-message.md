MECHANISM: Parameter-free final feature normalization

HYPOTHESIS: A 1,268-parameter model will retain at least 99% accuracy because the current 1,275-parameter model achieved 99.98%, while its mean-zero residual stream and tied learned embedding can represent the output geometry without seven independently learned final normalization scales.

INTENDED_EDIT: Replace the seven-parameter anchored final LayerNorm with parameter-free LayerNorm, preserving the current robust attention and MLP mechanisms while forcing output discrimination to be represented in the learned residual stream and tied embedding.

EVIDENCE: Positional sharing and hidden-bias reductions became fragile near 1,271 parameters, whereas the current design has substantially greater accuracy margin. All available designs assume learned anisotropic final rescaling is necessary; this cleanly tests the alternative that normalization should standardize features while the learned transformer and embedding carry the task representation.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class ParameterFreeLayerNorm(nn.Module):
    """Parameter-free normalization of the learned residual representation."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = AnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln_f = ParameterFreeLayerNorm(cfg.d_model)
>>>>>>> REPLACE