MECHANISM: Single-coordinate pre-attention scale-gauge anchoring

HYPOTHESIS: Fixing one `ln1` scale coordinate at its initialized value will produce a 1542-parameter model with at least 99% accuracy, because that scale can be absorbed into the corresponding unrestricted `qkv` input column while retaining seven learned normalization scales.

INTENDED_EDIT: Replace `ln1` with a LayerNorm whose final scale is fixed to one and whose other seven scales remain trainable, preserving the freshly initialized function.

EVIDENCE: The current 1543-parameter model achieved 99.83%; removing all eight `ln1` scales collapsed to 71.25%, motivating an incremental one-coordinate gauge reduction instead of repeating the all-at-once restriction.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with one scale fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE