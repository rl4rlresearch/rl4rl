MECHANISM: Six-dimensional factorized tied token embeddings

HYPOTHESIS: Restoring full eight-dimensional positional embeddings while factorizing only the token/input-output embedding through six learned dimensions will retain at least 99% accuracy and reduce parameters from 1,644 to 1,484.

INTENDED_EDIT: Replace the width-eight tied embedding/head with a learned width-six embedding projected into the unchanged width-eight residual stream and reuse its projected weights for logits.

EVIDENCE: Compressing positional embeddings to four dimensions reduced accuracy to 73.99%, showing that positional capacity must be preserved; the 99.96% baseline margin motivates testing compression in the distinct token-representation component instead.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_dim = max(1, cfg.d_model - 2)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.token_proj = nn.Linear(self.token_dim, cfg.d_model, bias=False)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        # The projected token weights are reused as the output classifier.
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.token_proj.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        x = self.token_proj(self.token_emb(idx)) + self.pos_emb(pos)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, self.token_proj(self.token_emb.weight))
>>>>>>> REPLACE