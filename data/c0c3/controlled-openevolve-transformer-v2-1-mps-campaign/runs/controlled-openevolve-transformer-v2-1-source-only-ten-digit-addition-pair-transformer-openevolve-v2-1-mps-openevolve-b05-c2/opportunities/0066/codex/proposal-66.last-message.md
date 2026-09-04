MECHANISM: Sixth pre-attention LayerNorm scale gauge

HYPOTHESIS: Reproducing the qualified query-bias and five-scale gauges, then fixing a sixth `ln1` scale, will yield a 1513-parameter model with at least 99% accuracy.

INTENDED_EDIT: Omit `q_bias[4]` as in the qualified 1514-parameter design and replace the four learned `ln1` scales with two learned scales and six fixed unit scales.

EVIDENCE: The query-bias plus fifth-scale design achieved 99.84% accuracy at 1514 parameters; extending that successful LayerNorm gauge by one coordinate is the closest untested reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
=======
        bsz, seqlen, d_model = x.shape
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
        qkv_bias = torch.cat(
            (
                q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuadrupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with four scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 4), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class SextupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with six scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 6), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = QuadrupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = SextupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE