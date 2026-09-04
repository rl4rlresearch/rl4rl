MECHANISM: Rank-6 shared embedding factorization

HYPOTHESIS: Constraining the tied token and positional embeddings to a learned six-dimensional subspace will retain at least 99% accuracy while reducing parameters from 1,644 to 1,418.

INTENDED_EDIT: Replace full-width token and positional tables with rank-6 learned codes projected into the existing eight-dimensional transformer, and reuse the resulting token embeddings for logits.

EVIDENCE: The current eight-dimensional model reaches 99.96% accuracy, leaving substantial margin for a two-dimensional embedding bottleneck without reducing attention or MLP capacity.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.embedding_rank = 6
        self.token_code = nn.Embedding(cfg.vocab_size, self.embedding_rank)
        self.pos_code = nn.Embedding(cfg.max_seq_len, self.embedding_rank)
        self.embed_proj = nn.Linear(self.embedding_rank, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(
            self.embed_proj.weight,
            mean=0.0,
            std=1.0 / math.sqrt(self.embedding_rank),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.embed_proj(self.token_code(idx) + self.pos_code(pos))
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        token_weight = self.embed_proj(self.token_code.weight)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE