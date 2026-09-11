MECHANISM: Third pre-attention LayerNorm bias quotient

HYPOTHESIS: Removing a third `ln1` bias coordinate will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because downstream query and value biases absorb its constant affine effects while the corresponding key shift is softmax-invariant.

INTENDED_EDIT: Reuse the unused LayerNorm variant as a three-coordinate-pruned LayerNorm and apply it only before attention, leaving `ln2` and all training settings unchanged.

EVIDENCE: The verified 1,554-parameter model already succeeds with two bias coordinates removed from `ln1`; unlike the failed `ln1` scale removal, another bias removal follows the same downstream-affine redundancy already tolerated by the successful design.

<<<<<<< SEARCH
class OnePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with one bias coordinate absorbed by downstream attention biases."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 1))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class ThreePrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with three bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-3].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 3))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = ThreePrunedBiasLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = TwoPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE