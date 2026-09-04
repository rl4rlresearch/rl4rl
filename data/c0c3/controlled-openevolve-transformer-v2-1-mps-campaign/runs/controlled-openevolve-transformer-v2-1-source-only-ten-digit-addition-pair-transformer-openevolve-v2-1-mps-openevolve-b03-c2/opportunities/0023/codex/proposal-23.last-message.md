MECHANISM: Downstream projection-bias absorption of pre-LayerNorm offsets

HYPOTHESIS: Removing both block LayerNorm bias vectors will produce a 1595-parameter model with at least 99% accuracy, because their affine offsets can be represented by the existing QKV and `fc1` biases without reducing the learned model’s function class.

INTENDED_EDIT: Disable the eight-parameter bias in `ln1` and `ln2` while retaining their learned scale parameters and all downstream projection biases.

EVIDENCE: The current 1611-parameter quotient design achieved 99.51% accuracy, and prior successful reductions show that exact redundant degrees can be removed while retaining accuracy. LayerNorm biases initialize to zero, while `qkv` and `fc1` already provide the downstream offsets needed to absorb them.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE