MECHANISM: Shared pre-sublayer normalization affine parameters

HYPOTHESIS: Tying `ln2.weight` to `ln1.weight` will reduce deduplicated parameters from 1,636 to 1,628 while retaining at least 99% accuracy because the validated shared-bias design indicates both normalization sites can share affine structure without removing attention or MLP capacity.

INTENDED_EDIT: Tie both the scale and bias parameters of the block’s two LayerNorm modules.

EVIDENCE: Sharing the two LayerNorm biases achieved 99.81% accuracy with 1,636 parameters, while narrowing the MLP or removing QKV biases failed; this motivates another isolated eight-parameter sharing constraint within the already validated normalization pair.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = self.ln1.bias
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.weight = self.ln1.weight
        self.ln2.bias = self.ln1.bias
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE