MECHANISM: Four-dimensional learned positional subspace

HYPOTHESIS: Because the 1,644-parameter baseline reaches 99.96% accuracy, constraining only its positional embeddings to four active dimensions will preserve at least 99% accuracy while reducing the model to 1,152 parameters.

INTENDED_EDIT: Replace each eight-dimensional positional embedding with a learned four-dimensional embedding, zero-padded to the unchanged eight-dimensional residual stream.

EVIDENCE: The current one-layer, width-eight model achieves 0.9996 accuracy at 1,644 parameters, indicating enough accuracy margin to compress positional representation without reducing attention or MLP capacity.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, min(4, cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        pos_x = F.pad(self.pos_emb(pos), (0, self.cfg.d_model - self.pos_emb.embedding_dim))
        x = self.token_emb(idx) + pos_x
>>>>>>> REPLACE