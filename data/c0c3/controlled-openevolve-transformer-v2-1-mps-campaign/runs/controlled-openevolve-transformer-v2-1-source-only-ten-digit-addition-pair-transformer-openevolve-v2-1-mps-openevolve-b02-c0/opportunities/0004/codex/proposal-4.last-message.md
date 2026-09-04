MECHANISM: One-coordinate pre-attention bias anchoring

HYPOTHESIS: Fixing one coordinate of the first LayerNorm bias at zero will reduce the model from 1,644 to 1,643 parameters while retaining at least 99% accuracy, because the following learned QKV bias can absorb the omitted constant offset.

INTENDED_EDIT: Add a LayerNorm equivalent with seven learned bias coordinates and one fixed-zero coordinate, and use it only before self-attention.

EVIDENCE: The 1,644-parameter model reached 99.96%, while larger structural reductions collapsed accuracy; this motivates the smallest possible reduction in a bias that is representationally redundant with the following affine projection.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
class AnchoredLayerNorm(nn.Module):
    """LayerNorm with one fixed-zero bias coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, (x.size(-1),), self.weight, bias, 1e-5)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AnchoredLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE