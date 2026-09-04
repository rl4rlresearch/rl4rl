MECHANISM: Pre-attention LayerNorm scale folding

HYPOTHESIS: Applying the qualified 1555-parameter design and removing `ln1`’s eight scale parameters will produce a 1547-parameter model with at least 99% accuracy, because the scale can be absorbed exactly into the unconstrained input columns of `qkv`.

INTENDED_EDIT: Fold the value bias into a full mean-zero attention projection bias, make both block LayerNorms affine-free, and retain the learned query bias.

EVIDENCE: The affine-free `ln2` design achieved 99.89% accuracy with 1555 parameters; `ln1` has the same learned-linear successor structure, while the qualified bias-free `ln1` result shows its additive shift is already unnecessary.

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = TwiceAnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
=======
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE