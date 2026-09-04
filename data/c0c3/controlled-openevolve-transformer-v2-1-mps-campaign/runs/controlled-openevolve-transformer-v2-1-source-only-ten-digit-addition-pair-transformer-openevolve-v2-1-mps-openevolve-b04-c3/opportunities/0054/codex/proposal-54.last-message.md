MECHANISM: Rank-six tied token code with shared query-bias gauge

HYPOTHESIS: Combining the qualified rank-six learned-position model with the independently qualified one-coordinate query-bias sharing will achieve at least 99% accuracy after 45,000 updates with 1,351 learned parameters.

INTENDED_EDIT: Replace full-width token embeddings and fixed sinusoidal positions with six-dimensional tied token codes and learned positions, retain value-bias removal and one-coordinate query-bias sharing, and validate only at the final step of a 45,000-step run.

EVIDENCE: The rank-six learned-position model reached 100% accuracy at 1,360 parameters, value-bias removal retained 100% at 1,352, and one-coordinate query-bias sharing independently reached 99.97%; this combines those orthogonal successful reductions.

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * frequencies)
        pos_encoding[:, 1::2] = torch.cos(
            positions * frequencies[: pos_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.ones(()))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg

        # Learn token identities in a six-dimensional subspace, then lift them
        # into the eight-dimensional residual stream without extra parameters.
        # The same learned code is used by the output classifier.
        self.token_dim = cfg.d_model - 2
        if self.token_dim <= 0:
            raise ValueError("d_model must be at least 3")
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = F.linear(
            x[..., : self.token_dim],
            self.token_scale * self.token_emb.weight,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE