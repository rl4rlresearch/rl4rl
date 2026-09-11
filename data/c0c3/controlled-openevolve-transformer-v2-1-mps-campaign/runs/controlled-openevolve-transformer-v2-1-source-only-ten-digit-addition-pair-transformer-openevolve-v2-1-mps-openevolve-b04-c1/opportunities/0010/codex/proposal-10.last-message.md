MECHANISM: LayerNorm-gauge-fixed positional codes

HYPOTHESIS: Representing each learned position with seven coordinates and padding an eighth zero will remove `max_seq_len` parameters while retaining at least 99% accuracy, because position-dependent shifts along the all-ones direction are invisible to every pre-LayerNorm sublayer and the final LayerNorm.

INTENDED_EDIT: Decouple positional-code width from the load-bearing eight-dimensional residual stream by using a seven-dimensional learned positional embedding with a fixed zero gauge coordinate.

EVIDENCE: Reducing the entire residual width to six collapsed accuracy to 0.01%, while the eight-wide model reaches 99.91%. Prior designs assumed positional and computational widths must match; this patch preserves all attention and MLP capacity while removing a functionally redundant positional degree of freedom.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        x = self.token_emb(idx) + F.pad(self.pos_emb(pos), (0, 1))
>>>>>>> REPLACE