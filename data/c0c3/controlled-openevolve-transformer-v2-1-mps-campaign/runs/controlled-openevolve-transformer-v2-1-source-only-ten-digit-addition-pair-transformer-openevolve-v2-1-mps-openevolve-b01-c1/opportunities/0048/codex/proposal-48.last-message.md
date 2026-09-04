MECHANISM: Zero-beta LayerNorm scale gauge

HYPOTHESIS: Fixing one `ln1` scale coordinate at 1 will reduce the model from 1,578 to 1,577 parameters while retaining at least 99% accuracy, because that coordinate’s bias is already fixed at zero and the following QKV weight column can absorb its learned scale without reducing the attention function family or changing initialization outputs.

INTENDED_EDIT: Add a two-bias-pruned LayerNorm with one fixed unit scale coordinate and use it only for `ln1`, leaving the successful `ln2` and one-pruned `fc1` pathway unchanged.

EVIDENCE: The one-pruned `fc1` design achieved 99.92% at 1,578 parameters, while removing a second `fc1` bias fell to 84.48%; this motivates a single exact gauge removal elsewhere that preserves the six learned `ln2` bias coordinates implicated in the successful result.

<<<<<<< SEARCH
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with seven bias directions absorbed by token-row means."""
=======
class OnePrunedScaleTwoPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with two zero biases and one scale absorbed by a downstream column."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()
        self.weight = nn.Parameter(full_weight[:-1].clone())
        self.bias = nn.Parameter(full_bias[:-2].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        bias = F.pad(self.bias, (0, 2))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with seven bias directions absorbed by token-row means."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = OnePrunedScaleTwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE