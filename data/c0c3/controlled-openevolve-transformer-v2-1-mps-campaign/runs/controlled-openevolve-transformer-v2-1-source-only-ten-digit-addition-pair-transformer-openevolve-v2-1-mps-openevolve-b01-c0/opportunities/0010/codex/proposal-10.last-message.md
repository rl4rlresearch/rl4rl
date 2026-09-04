MECHANISM: Locally absorbable pre-MLP normalization bias

HYPOTHESIS: Fixing one `ln2` bias coordinate at zero will reduce the model from 1636 to 1635 parameters while retaining at least 99% accuracy, because its effect is exactly representable by the full learned `fc1` bias and the initial model computation remains unchanged.

INTENDED_EDIT: Replace the block’s second LayerNorm with an equivalent implementation containing seven learned bias coordinates and one fixed-zero coordinate, while retaining all gains and MLP parameters.

EVIDENCE: The 1636-parameter design reached 99.98%, while narrowing the MLP collapsed accuracy and attention-path bias reductions proved optimization-sensitive; this tests a new, strictly local redundancy immediately before the MLP’s learned affine bias without reducing MLP width.

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
class ReducedBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = ReducedBiasLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE