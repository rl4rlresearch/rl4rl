MECHANISM: Single positional-coordinate pruning

HYPOTHESIS: Fixing one positional-embedding scalar to zero will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store the learned non-anchor positional table with its final scalar omitted, reconstruct that scalar as zero, and otherwise preserve initialization and training.

EVIDENCE: The current 1,269-parameter model reached 100%, while substantially stronger positional compression previously reached 97.63%. A one-scalar positional restriction is the smallest alternative to the additional attention gauge constraints that repeatedly disrupted optimization at 1,268 parameters.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        self.pos_emb = nn.Parameter(
            torch.empty((cfg.max_seq_len - 1) * (cfg.d_model - 1) - 1)
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
=======
        learned_pos_weight = F.pad(self.pos_emb, (0, 1)).view(
            self.cfg.max_seq_len - 1, self.cfg.d_model - 1
        )
        pos_weight = F.pad(learned_pos_weight, (0, 0, 1, 0))
>>>>>>> REPLACE