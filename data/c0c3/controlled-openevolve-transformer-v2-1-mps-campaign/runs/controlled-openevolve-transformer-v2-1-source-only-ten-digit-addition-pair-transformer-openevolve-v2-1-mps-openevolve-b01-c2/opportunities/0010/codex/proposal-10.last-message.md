MECHANISM: Isolated attention-side normalization scale removal

HYPOTHESIS: Removing only `ln1`’s scale while retaining `ln2`’s scale will reduce the model to 1,604 parameters and maintain at least 99% accuracy, because the attention QKV weights can absorb `ln1`’s multiplicative scale while the MLP retains its normalization scale.

INTENDED_EDIT: Disable all affine parameters in `ln1` and remove only the bias from `ln2`.

EVIDENCE: Removing both LayerNorm biases achieved 99.95% with 1,612 parameters, while removing both scales simultaneously fell to 79.63%; isolating one scale tests whether the failure came from eliminating both optimization pathways together.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        # The QKV input weights can absorb ln1's scale; retain ln2's scale to
        # preserve the MLP optimization pathway implicated by the joint failure.
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE