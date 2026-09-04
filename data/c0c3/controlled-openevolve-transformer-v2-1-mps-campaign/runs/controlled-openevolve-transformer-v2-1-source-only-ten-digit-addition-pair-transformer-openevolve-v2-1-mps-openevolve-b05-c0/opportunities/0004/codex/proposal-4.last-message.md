MECHANISM: Remove affine offsets duplicated by downstream biases

HYPOTHESIS: Disabling both LayerNorm bias vectors will reduce the model from 1636 to 1620 parameters while retaining at least 99% accuracy, because the MLP LayerNorm offset is absorbable by `fc1.bias`, while the attention LayerNorm offset is absorbable by the retained query/value biases and produces only a softmax-invariant constant key shift.

INTENDED_EDIT: Keep both learned LayerNorm scale vectors but construct the two block LayerNorms without learned biases.

EVIDENCE: The 1636-parameter design reaches 99.84% accuracy while preserving query and value biases; removing value bias caused optimization failure, so this patch retains those successful parameters and instead removes upstream offsets that duplicate them and the existing MLP input bias.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE