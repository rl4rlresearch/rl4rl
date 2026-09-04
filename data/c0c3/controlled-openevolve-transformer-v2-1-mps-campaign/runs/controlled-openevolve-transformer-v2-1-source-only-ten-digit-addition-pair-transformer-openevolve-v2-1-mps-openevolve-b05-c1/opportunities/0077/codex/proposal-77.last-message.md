MECHANISM: Gauge-fixed tied token embeddings

HYPOTHESIS: Fixing one token-embedding entry to zero will reduce the model from 1,273 to 1,272 parameters while preserving at least 99% accuracy, because subtracting that entry uniformly from every token-embedding coordinate changes residual inputs only by a LayerNorm-invisible all-ones shift and changes every output logit by the same amount.

INTENDED_EDIT: Store the tied token-embedding matrix in a one-parameter-smaller chart whose first entry is fixed at zero, reconstruct it for both input lookup and output projection, and otherwise leave the model and training unchanged.

EVIDENCE: Anchoring position 0 by removing the exact positional/token-embedding translation redundancy retained 100% accuracy at 1,273 parameters; this applies the remaining scalar translation symmetry of the tied token embeddings without restricting observable model functions.

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = FixedBiasLayerNorm(cfg.d_model)

        self.apply(self._init_weights)
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, seqlen = idx.shape
=======
    def token_weight(self) -> torch.Tensor:
        return F.pad(self.token_emb, (1, 0)).view(
            self.cfg.vocab_size, self.cfg.d_model
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        _, seqlen = idx.shape
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = self.token_emb(idx) + F.pad(F.embedding(pos, pos_weight), (0, 1))
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_weight())
>>>>>>> REPLACE