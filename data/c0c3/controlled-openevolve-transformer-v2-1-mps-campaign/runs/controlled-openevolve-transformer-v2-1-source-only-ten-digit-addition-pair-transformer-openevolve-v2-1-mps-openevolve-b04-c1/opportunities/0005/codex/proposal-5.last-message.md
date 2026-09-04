MECHANISM: Remove downstream-absorbed pre-sublayer normalization biases

HYPOTHESIS: Removing the two 8-element LayerNorm biases will reduce parameters from 1636 to 1620 while retaining at least 99% accuracy, because the attention LayerNorm bias is absorbable by the query/value biases with its key component softmax-invariant, and the MLP LayerNorm bias is absorbable by `fc1.bias`.

INTENDED_EDIT: Disable only the bias parameters in `ln1` and `ln2`, preserving their learned scales and every other architectural and training setting.

EVIDENCE: The 1636-parameter design reached 99.85% after removing the exactly redundant key bias, while broad bias removal failed; this motivates removing only two further biases with explicit downstream reparameterizations.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE