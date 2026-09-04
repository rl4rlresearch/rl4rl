MECHANISM: Single-position common-mode gauge fixing

HYPOTHESIS: Fixing only one positional-embedding coordinate will reduce the verified model from 1,315 to 1,314 parameters while retaining at least 99% accuracy within 50,000 steps, because a common-mode shift of that position is erased by every downstream LayerNorm.

INTENDED_EDIT: Store all but one positional-embedding coordinate as learned parameters, append one fixed-zero coordinate, and reconstruct the positional table for embedding lookup.

EVIDENCE: Quotienting every positional row reduced accuracy to 69.6%, indicating optimization sensitivity to removing `INPUT_LEN` redundant directions at once; the 1,315-parameter design achieved 100%, so removing only one exact LayerNorm-invariant direction is the smallest informative test.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len * cfg.d_model - 1, 1)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight.flatten(), (0, 1)).view(
            self.cfg.max_seq_len, self.cfg.d_model
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)
>>>>>>> REPLACE