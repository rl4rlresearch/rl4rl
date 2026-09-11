MECHANISM: Incremental attention LayerNorm bias absorption

HYPOTHESIS: Fixing a third `ln1` bias coordinate to zero will reduce parameters from 1,593 to 1,592 while retaining at least 99% accuracy, because the downstream QKV biases can absorb the omitted coordinate.

INTENDED_EDIT: Add a three-coordinate-pruned LayerNorm and use it only before causal self-attention, leaving the successful two-coordinate `ln2` pruning and all other settings unchanged.

EVIDENCE: Two-coordinate `ln1` pruning achieved 99.88% at 1,593 parameters; although third-coordinate `ln2` pruning reached only 98.12%, testing the same smallest incremental reduction in the still-successful attention path is the most direct next experiment.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
class ThreePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with three bias coordinates absorbed by downstream QKV biases."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = ThreePrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE