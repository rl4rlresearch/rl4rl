MECHANISM: Third zero-initialized query-bias anchor with the qualified adaptive LayerNorm scale

HYPOTHESIS: Reproducing the qualified 1511-parameter design and fixing `q_bias[1]` will yield a 1510-parameter model with at least 99% accuracy.

INTENDED_EDIT: Fix seven `ln1` scales while retaining its sole learned scale, and reconstruct query-bias coordinates 0, 1, and 4 as zero.

EVIDENCE: The independent `ln1` scale plus `q_bias[0]`/`q_bias[4]` design achieved 99.96% at 1511 parameters; the failed 1510 design shared that essential scale, so this tests a distinct query-bias reduction while leaving the scale independent.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(2),
                self.q_bias[:self.head_dim - 2],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuintupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with five scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 5), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = QuintupleAnchoredScaleLayerNorm(cfg.d_model)
=======
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


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE