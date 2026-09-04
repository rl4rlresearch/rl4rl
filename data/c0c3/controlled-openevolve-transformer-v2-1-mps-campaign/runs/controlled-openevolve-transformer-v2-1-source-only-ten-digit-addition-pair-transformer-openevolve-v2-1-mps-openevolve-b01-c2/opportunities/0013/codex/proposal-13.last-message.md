MECHANISM: Residual common-mode bias gauge fixing

HYPOTHESIS: Combining the verified single-coordinate `ln2` scale anchor with a fixed zero coordinate in the attention output bias will produce a 1,610-parameter model with at least 99% accuracy after 5,000 steps.

INTENDED_EDIT: Retain seven learned `ln2` scales, retain seven learned attention projection-bias coordinates, and synthesize fixed unit-scale and zero-bias coordinates during forward passes.

EVIDENCE: The one-coordinate `ln2` anchor achieved 100% accuracy with 1,611 parameters, while a second scale anchor fell to 73.4%; removing an orthogonal additive gauge is therefore better motivated. A uniform shift in the attention projection bias is invisible to `ln2` and can be absorbed exactly by the MLP output bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
        # Fix one projection-bias coordinate. Any discarded common-mode
        # component is invisible to ln2 and can be absorbed by fc2.bias.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = torch.cat((self.proj.bias, self.proj.bias.new_zeros(1)))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Pre-sublayer normalization offsets are absorbed by downstream biases.
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
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
        # Pre-sublayer normalization offsets are absorbed by downstream biases.
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE