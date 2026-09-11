MECHANISM: Distributed single-coordinate LayerNorm scale anchoring

HYPOTHESIS: Fixing one scale coordinate in each pre-sublayer LayerNorm will produce a 1,610-parameter model with at least 99% accuracy, because each normalization retains seven adaptive scales and its downstream learned projection can absorb the fixed coordinate.

INTENDED_EDIT: Replace both biased LayerNorms with bias-free LayerNorms that learn seven scale coordinates and synthesize one fixed unit scale.

EVIDENCE: A single anchored `ln2` coordinate achieved 100% accuracy with 1,611 parameters, while anchoring a second `ln2` coordinate failed at 73.4%; distributing the second anchor to `ln1` preserves seven adaptive coordinates in both pathways. Removing all `ln1` scales failed, motivating this one-coordinate reduction instead.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE