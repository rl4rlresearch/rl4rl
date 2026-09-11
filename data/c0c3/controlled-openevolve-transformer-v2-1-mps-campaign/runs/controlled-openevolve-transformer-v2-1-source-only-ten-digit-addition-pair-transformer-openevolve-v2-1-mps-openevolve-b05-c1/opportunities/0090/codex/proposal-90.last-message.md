MECHANISM: Learned rank-7 token/unembedding geometry

HYPOTHESIS: Replacing the independent 8D token table with a well-conditioned learned rank-7 factorization will retain at least 99% accuracy while reducing the verified model from 1,269 to 1,212 parameters.

INTENDED_EDIT: Factor the tied token embedding and output classifier into learned per-token 7D codes and a learned 7×8 basis, initialized as an orthonormal centered subspace.

EVIDENCE: Prior positional compression and even one-scalar positional pruning failed, indicating the learned absolute-position table is load-bearing. This instead challenges the assumption that all 114 tokens require independent 8D geometry; LayerNorm exposes seven centered contextual degrees, while the learned basis can rotate away from its centered initialization if the final classifier needs another orientation.

<<<<<<< SEARCH
        self.token_emb = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        self.token_codes = nn.Parameter(
            torch.empty(cfg.vocab_size, cfg.d_model - 1)
        )
        self.token_basis = nn.Parameter(
            torch.empty(cfg.d_model - 1, cfg.d_model)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(self.token_codes, mean=0.0, std=0.02)
        with torch.no_grad():
            seed_basis = torch.eye(
                cfg.d_model,
                dtype=self.token_basis.dtype,
                device=self.token_basis.device,
            )[:-1]
            seed_basis = seed_basis - seed_basis.mean(dim=1, keepdim=True)
            orthonormal = torch.linalg.qr(
                seed_basis.transpose(0, 1),
                mode="reduced",
            ).Q.transpose(0, 1)
            self.token_basis.copy_(orthonormal)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def token_weight(self) -> torch.Tensor:
        return F.pad(self.token_emb, (1, 0)).view(
            self.cfg.vocab_size, self.cfg.d_model
        )
=======
    def token_weight(self) -> torch.Tensor:
        return self.token_codes @ self.token_basis
>>>>>>> REPLACE