MECHANISM: Endpoint final-normalization bias anchoring

HYPOTHESIS: Reproducing the qualified 1495-parameter design and fixing final LayerNorm bias coordinate 7 while preserving coordinates 1–6 will yield 1494 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Fix all pre-attention LayerNorm scales at one, then reconstruct final LayerNorm bias coordinates 0 and 7 as zero while learning all eight scales and bias coordinates 1–6.

EVIDENCE: Fixing final bias coordinate 0 achieved 99.83% accuracy at 1495 parameters; additional anchors at coordinates 1–6 all failed, so coordinate 7 is the only untested continuation in this successful zero-initialized bias family.

<<<<<<< SEARCH
class SeptupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with seven scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 7), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class FixedScaleLayerNorm(nn.Module):
    """LayerNorm with all scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class EndpointBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinates 0 and 7 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 1))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = EndpointBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE