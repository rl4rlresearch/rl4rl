MECHANISM: Single-coordinate MLP LayerNorm bias quotient

HYPOTHESIS: Fixing one coordinate of the second pre-LayerNorm bias to zero will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because its effect can be absorbed exactly into the retained `fc1` bias.

INTENDED_EDIT: Add a LayerNorm variant with seven learned bias coordinates and use it only before the MLP, reconstructing the eighth coordinate as zero without changing initialization RNG consumption.

EVIDENCE: The current mean-free positional design achieved 99.64% at 1,619 parameters, while broader removals caused optimization failures; this tests the smallest possible reduction in a directly redundant affine parameter.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class SinglePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with one bias coordinate absorbed by the following linear bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = SinglePrunedBiasLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE