MECHANISM: Final pre-attention LayerNorm scale absorption

HYPOTHESIS: Starting from the qualified 1506-parameter `q_bias[6]`-anchored design, fixing the remaining pre-attention LayerNorm scale at one will yield 1505 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified `q_bias[6]` anchor, then make `ln1` fully parameter-free by fixing its final learned scale, which the following learned QKV map can absorb.

EVIDENCE: Anchoring `q_bias[6]` achieved 99.78% at 1506 parameters, while removing `q_bias[7]` fell to 94.48% and additional value rotations also failed; all qualified designs already fix seven of eight `ln1` scales, motivating completion of that distinct gauge.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 3:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[
                    self.head_dim - 3:self.head_dim - 2
                ],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
        )
>>>>>>> REPLACE

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


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
class FullyAnchoredScaleLayerNorm(nn.Module):
    """Parameter-free LayerNorm with every scale absorbed by the QKV map."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = FullyAnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE