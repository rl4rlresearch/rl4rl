MECHANISM: Shared sinusoidal absolute-position manifold

HYPOTHESIS: Replacing independent position vectors with a fixed sinusoidal basis and eight learned channel scales will reduce parameters by `8 * (max_seq_len - 1)` while retaining at least 99% accuracy, because addition’s position relationships are regular and can be represented compositionally without narrowing the successful attention or MLP.

INTENDED_EDIT: Remove the learned positional embedding table; encode absolute positions with generic sinusoidal features whose channel amplitudes are learned from fresh initialization.

EVIDENCE: The 1,596-parameter `d_model=8, d_ff=12` model reached 99.99%, while narrower MLPs and small affine gauge changes failed. This challenges the shared assumption that every position needs an independent learned vector while preserving the demonstrated load-bearing feature capacity.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        position = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        pos_basis = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_basis[:, 0::2] = torch.sin(position * frequency)
        pos_basis[:, 1::2] = torch.cos(
            position * frequency[: pos_basis[:, 1::2].shape[1]]
        )
        self.register_buffer("pos_basis", pos_basis, persistent=False)
        self.pos_scale = nn.Parameter(torch.full((cfg.d_model,), 0.02))

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = self.pos_basis[:seqlen].unsqueeze(0)
        x = self.token_emb(idx) + pos * self.pos_scale
        x = self.drop(x)
>>>>>>> REPLACE