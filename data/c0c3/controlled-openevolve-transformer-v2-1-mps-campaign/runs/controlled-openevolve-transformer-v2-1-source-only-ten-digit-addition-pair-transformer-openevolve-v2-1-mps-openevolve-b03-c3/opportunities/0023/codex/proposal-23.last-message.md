MECHANISM: Pre-attention bias absorption

HYPOTHESIS: Making both block LayerNorms bias-free will yield 1,620 parameters and retain at least 99% accuracy, because the retained query/value biases can absorb the first LayerNorm’s query/value offsets while its position-independent key offset cancels in attention softmax.

INTENDED_EDIT: Remove the learned biases from `ln1` and `ln2` while retaining query and value attention biases and the full width-12 MLP.

EVIDENCE: Removing `ln2` bias while retaining query/value biases achieved 99.95% at 1,628 parameters; the query-bias ablation failed at 48.92%, motivating preservation of both attention biases while eliminating the redundant upstream bias instead.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE