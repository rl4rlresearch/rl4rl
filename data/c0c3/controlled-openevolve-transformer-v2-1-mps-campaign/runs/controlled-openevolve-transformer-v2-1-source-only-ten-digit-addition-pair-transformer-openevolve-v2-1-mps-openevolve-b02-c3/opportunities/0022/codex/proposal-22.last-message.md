MECHANISM: Quotient-null token-subspace elimination

HYPOTHESIS: Removing the seven token-subspace tilt parameters from the qualified relative-position design will yield a 1,325-parameter transformer with at least 99% accuracy, because every residual and positional contribution is zero-mean, LayerNorm removes the token embedding’s all-ones component, and the tied classifier receives a zero-mean state.

INTENDED_EDIT: Apply the verified 1,332-parameter sinusoidal/relative-distance design, then fix the token projection to its seven-dimensional zero-mean basis instead of learning seven functionally unobservable tilt parameters.

EVIDENCE: Reference Design 2 achieved 99.96% accuracy with 1,332 parameters. Its token tilt starts at zero, while its fixed zero-mean positional encoding, quotient residual projections, and final LayerNorm make the tilt’s all-ones component unobservable throughout the model.

<<<<<<< SEARCH
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)
        self.tilt = nn.Parameter(torch.zeros(rank))

    def projection_weight(self) -> torch.Tensor:
        return self.basis + torch.outer(self.normal, self.tilt)
=======
        self.register_buffer("basis", basis, persistent=False)

    def projection_weight(self) -> torch.Tensor:
        return self.basis
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.norm = nn.LayerNorm(normalized_shape, bias=False)
=======
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = QuotientOutputLinear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        relative_bias = torch.cat(
            (
                self.relative_bias.new_zeros(self.n_head, 1),
                self.relative_bias[:, : seqlen - 1],
            ),
            dim=1,
        )
        att = att + relative_bias[:, distance.clamp_min(0)].unsqueeze(0)

        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = QuotientPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            / cfg.d_model
        )
        angles = positions * frequencies.unsqueeze(0)
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(angles)
        odd_features = pos_encoding[:, 1::2].shape[1]
        pos_encoding[:, 1::2] = torch.cos(angles[:, :odd_features])
        pos_encoding = pos_encoding - pos_encoding.mean(dim=-1, keepdim=True)
        self.register_buffer("pos_encoding", pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.tensor(0.02))

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        x = self.token_emb(idx)
        x = x + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE