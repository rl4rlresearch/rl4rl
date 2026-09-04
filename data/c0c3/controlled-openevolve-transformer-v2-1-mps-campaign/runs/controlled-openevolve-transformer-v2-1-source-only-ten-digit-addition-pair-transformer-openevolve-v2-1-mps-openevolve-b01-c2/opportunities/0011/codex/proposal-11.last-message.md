MECHANISM: Single-coordinate LayerNorm scale gauge fixing

HYPOTHESIS: Fixing one `ln2` scale coordinate at its initial value while retaining the other seven learned scales will produce a 1,611-parameter model with at least 99% accuracy, because the following MLP input weight can absorb that coordinate’s scale without removing the adaptive normalization pathway.

INTENDED_EDIT: Apply the verified query-only QKV bias and bias-free pre-LayerNorm design, then replace `ln2` with a bias-free LayerNorm having seven learned scales and one fixed unit scale.

EVIDENCE: The 1,612-parameter query-only, bias-free LayerNorm design achieved 99.95%; removing all scales failed, so anchoring only one MLP-side scale is a conservative exact reparameterization that preserves seven adaptive scales.

<<<<<<< SEARCH
        # Construct the baseline projection first to preserve its RNG stream, then
        # retain only the effective query and value bias parameters.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
=======
        # Preserve fused projection construction while retaining only query bias.
        # Key bias cancels in softmax and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qv_bias = self.qkv.bias
        fused_bias = torch.cat(
            (qv_bias[:d_model], qv_bias.new_zeros(d_model), qv_bias[d_model:])
        )
=======
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
>>>>>>> REPLACE

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
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE