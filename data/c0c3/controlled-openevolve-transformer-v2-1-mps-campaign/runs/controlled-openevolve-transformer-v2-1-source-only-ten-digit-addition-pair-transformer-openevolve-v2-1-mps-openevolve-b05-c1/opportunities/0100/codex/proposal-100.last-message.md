MECHANISM: Learned relative-distance attention routing

HYPOTHESIS: Replacing independent absolute-position embeddings with per-head learned relative-distance attention biases will retain at least 99% accuracy while reducing the model from 1,266 to 1,156 parameters, because aligned addition columns and preceding outputs can be selected by relative distance without encoding every absolute position in the residual stream.

INTENDED_EDIT: Remove the 154-parameter absolute-position table and add a 44-parameter per-head causal relative-bias table, with one softmax-redundant bias fixed per head.

EVIDENCE: Strong positional compression previously reached 97.63%, showing that the full absolute table is not entirely necessary, while single-scalar pruning was optimization-destructive. This tests a different learned computation: positional routing occurs directly in attention rather than through compressed absolute embeddings.

<<<<<<< SEARCH
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        positions = torch.arange(max_seq_len)
        relative_distance = (
            positions[:, None] - positions[None, :]
        ).clamp_min(0)
        self.register_buffer(
            "relative_distance", relative_distance, persistent=False
        )
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        relative_distance = self.relative_distance[:seqlen, :seqlen]
        relative_bias = F.pad(self.rel_bias, (1, 0))[:, relative_distance]
        att = (
            (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            + relative_bias
        )
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
        x = self.drop(x)
=======
        x = self.drop(F.embedding(idx, self.token_weight()))
>>>>>>> REPLACE