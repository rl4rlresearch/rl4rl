MECHANISM: Learned-scale sinusoidal positional representation

HYPOTHESIS: Replacing the learned absolute-position table with a fixed multi-frequency sinusoidal basis and one learned gain will retain at least 99% accuracy after 75,000 steps while removing `8 * max_seq_len - 1` learned parameters.

INTENDED_EDIT: Replace per-position learned vectors with generic sinusoidal features scaled by one learned parameter, and extend training from 50,000 to 75,000 steps.

EVIDENCE: Every qualified design retains a separate learned vector for every sequence position, while the current 1,580-parameter model reaches 99.93% accuracy. This tests whether the learned attention and MLP can instead route information through a compact relative-offset-friendly positional basis, targeting substantially more parameters than further bias removal.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # A generic Fourier basis replaces the independently learned vector at
        # every position. Attention learns how to use these positional features,
        # while one gain adapts their magnitude relative to token embeddings.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=50000)
=======
    p.add_argument("--train-steps", type=int, default=75000)
>>>>>>> REPLACE