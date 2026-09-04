MECHANISM: Cross-sublayer LayerNorm gain sharing

HYPOTHESIS: Sharing the remaining attention LayerNorm gain with MLP LayerNorm gain coordinate 1 will produce a 1575-parameter model with at least 99% accuracy, because both gains are downstream-absorbable gauge scalars while the gain remains learned rather than frozen.

INTENDED_EDIT: Rebind the attention LayerNorm gain parameter to the MLP LayerNorm’s two-coordinate gain parameter; its first coordinate controls both norms, while the second remains exclusive to the MLP norm.

EVIDENCE: Fixing attention LayerNorm gain coordinate 3 achieved 99.92% at 1576 parameters, but fixing MLP LayerNorm gain coordinate 1 collapsed to 15.25%. Sharing removes the independent redundant scalar while preserving learned gain adaptation, directly testing whether the failure was caused by freezing that gain.

<<<<<<< SEARCH
        self.ln1 = AttentionAnchoredLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = MLPAnchoredLayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = AttentionAnchoredLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = MLPAnchoredLayerNorm(cfg.d_model)

        # The first MLP gain and the remaining attention gain are independent
        # downstream-absorbable gauges, so learn one scalar for both.
        self.ln1.weight_rest = self.ln2.weight_rest

        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE