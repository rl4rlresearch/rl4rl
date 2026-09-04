MECHANISM: Head-width learned positional bottleneck

HYPOTHESIS: Compressing each learned positional vector from 8 dimensions to the 4-dimensional attention-head width and broadcasting it across heads will retain at least 99% accuracy while removing `max_seq_len * 4` parameters.

INTENDED_EDIT: Store one head-width positional embedding per sequence position and repeat it across the model dimension before adding token embeddings.

EVIDENCE: The current 1-layer, 8-dimensional model reaches 99.96% accuracy; its two attention heads each operate on only four dimensions, motivating a rank-4 positional representation while preserving the successful attention and MLP capacity.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_width = cfg.d_model // cfg.n_head
        self.pos_emb = nn.Embedding(cfg.max_seq_len, self.pos_width)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        position = self.pos_emb(pos).repeat(1, 1, self.cfg.n_head)
        x = self.token_emb(idx) + position
>>>>>>> REPLACE