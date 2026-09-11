MECHANISM: Gauge-fixed positional embeddings

HYPOTHESIS: Removing one redundant common-mode coordinate from every positional embedding will reduce the model from 1,303 to 1,280 parameters while retaining at least 99% accuracy, because all downstream LayerNorms are invariant to per-position shifts along the all-ones direction.

INTENDED_EDIT: Learn an independent seven-coordinate embedding for each position and append a fixed-zero eighth coordinate before adding it to the token embedding.

EVIDENCE: Structured sinusoidal positions failed despite full dimensionality, indicating that independent positional freedom matters; this patch preserves seven independent degrees of freedom per position and removes only the common-shift degree that LayerNorm cannot observe.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        x = self.token_emb(idx) + F.pad(self.pos_emb(pos), (0, 1))
>>>>>>> REPLACE