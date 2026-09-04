MECHANISM: Redundant first pre-norm offset removal

HYPOTHESIS: Making `ln1` bias-free will reduce the model from 1,550 to 1,542 parameters while retaining at least 99% accuracy, because its query offset is representable by the retained learned query bias, its key offset cancels under softmax, and its position-independent value offset is representable by the attention output bias.

INTENDED_EDIT: Disable only the eight-dimensional bias of the first pre-attention LayerNorm while preserving its learned scale and all existing attention, MLP, and gauge parameters.

EVIDENCE: The current bias-free-value shared-key/value model achieves 99.85% at 1,550 parameters. Unlike the failed removal of `ln2` scales, this removes a zero-initialized affine offset whose three downstream effects already have exact absorption or invariance mechanisms, making it a controlled test at 1,542 parameters.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE