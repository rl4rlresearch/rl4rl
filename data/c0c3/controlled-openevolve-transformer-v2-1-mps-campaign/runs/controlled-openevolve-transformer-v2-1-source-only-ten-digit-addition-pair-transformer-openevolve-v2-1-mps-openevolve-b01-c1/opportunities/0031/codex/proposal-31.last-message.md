MECHANISM: Downstream-absorbed LayerNorm scale fixing

HYPOTHESIS: Fixing one `ln2` scale coordinate at its initial value will reduce the model from 1,585 to 1,584 parameters while retaining at least 99% accuracy, because the following `fc1` weight column can represent the removed multiplicative degree of freedom.

INTENDED_EDIT: Add a LayerNorm with two pruned bias coordinates and one fixed scale coordinate, and use it only before the MLP.

EVIDENCE: The current 1,585-parameter model achieved 99.97%, while further additive-bias and embedding-gauge pruning was optimization-sensitive. This edit preserves the initial function exactly and removes a local multiplicative redundancy that `fc1` can absorb.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class OneFixedScaleTwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with one scale fixed for absorption by the following linear."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = OneFixedScaleTwoPrunedBiasLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE