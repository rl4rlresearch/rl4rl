MECHANISM: Pre-attention LayerNorm bias folding

HYPOTHESIS: Removing the eight-parameter `ln1` bias will produce a 1563-parameter model with at least 99% accuracy because its query shift is representable by `q_bias`, its key shift cancels in softmax, and its value shift folds into the learned projection bias.

INTENDED_EDIT: Make the pre-attention LayerNorm bias-free while retaining the full query and mean-zero attention projection biases.

EVIDENCE: The current 1571-parameter model achieved 99.95%, and attention value/output-bias folding achieved 99.98%; the qualified designs also successfully use a bias-free `ln2`, supporting removal of an affine bias whose effects remain representable downstream.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE