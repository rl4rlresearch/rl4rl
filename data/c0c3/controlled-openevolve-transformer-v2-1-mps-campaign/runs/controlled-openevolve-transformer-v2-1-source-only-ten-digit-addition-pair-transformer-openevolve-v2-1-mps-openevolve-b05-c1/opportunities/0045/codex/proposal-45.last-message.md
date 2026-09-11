MECHANISM: Fixed-last-coordinate positional quotient

HYPOTHESIS: Representing each positional embedding with seven learned coordinates and a fixed-zero eighth coordinate will reduce the model by `INPUT_LEN` parameters while retaining at least 99% accuracy within 50,000 steps, because position-dependent common-mode shifts are erased by every LayerNorm.

INTENDED_EDIT: Reduce positional embeddings from eight to seven learned coordinates and pad the omitted coordinate with zero during the forward pass.

EVIDENCE: The current 1,376-parameter model achieved 100% accuracy at 50,000 steps, and a more restrictive rank-six positional factorization previously achieved 99.98%; this seven-coordinate chart removes only the exact LayerNorm-invariant common mode and adds one small padding operation.

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
        x = self.drop(x)
=======
        x = self.token_emb(idx) + F.pad(self.pos_emb(pos), (0, 1))
        x = self.drop(x)
>>>>>>> REPLACE