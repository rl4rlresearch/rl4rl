MECHANISM: Rank-seven tied vocabulary interface

HYPOTHESIS: Constraining the shared token-embedding/output matrix to learned rank seven will reduce the verified four-query-bias design from 1601 to 1543 parameters while retaining at least 99% accuracy, because the transformer’s eight-channel internal computation remains intact and only the least energetic singular direction of the tied vocabulary interface is removed.

INTENDED_EDIT: Replace the full tied vocabulary matrix with learned rank-seven token codes and basis factors, initialized from the best rank-seven approximation of the same fresh baseline draw; retain the qualified positional quotient and four-coordinate query bias.

EVIDENCE: The four-query-bias design achieved 99.96% at 1601 parameters, while several one-coordinate attention, LayerNorm, and MLP ablations failed sharply. This indicates those internal paths are load-bearing and motivates challenging the previously untouched assumption that the 114-by-8 tied vocabulary interface must be full rank.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class LowRankTokenEmbedding(nn.Embedding):
    """A learned rank-(d-1) vocabulary matrix shared with the output head."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the baseline embedding-constructor RNG before replacing its
        # full matrix with two learned low-rank factors.
        super().__init__(num_embeddings, embedding_dim)
        del self.weight
        self.rank = embedding_dim - 1
        self.codes = nn.Parameter(torch.empty(num_embeddings, self.rank))
        self.basis = nn.Parameter(torch.empty(self.rank, embedding_dim))

    @torch.no_grad()
    def initialize_from_full(self, full: torch.Tensor) -> None:
        # A balanced truncated SVD preserves as much of the corresponding
        # freshly drawn baseline vocabulary matrix as rank seven permits.
        u, singular, vh = torch.linalg.svd(full, full_matrices=False)
        scale = singular[: self.rank].sqrt()
        self.codes.copy_(u[:, : self.rank] * scale)
        self.basis.copy_(scale[:, None] * vh[: self.rank])

    def full_weight(self) -> torch.Tensor:
        return self.codes @ self.basis

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class LowRankTiedHead(nn.Linear):
    """Parameter-free head reusing a low-rank token embedding."""

    def __init__(self, token_embedding: LowRankTokenEmbedding):
        # Match the baseline head-constructor RNG stream, then discard its
        # independent matrix because logits use the learned shared factors.
        super().__init__(
            token_embedding.embedding_dim,
            token_embedding.num_embeddings,
            bias=False,
        )
        del self.weight
        object.__setattr__(self, "token_embedding", token_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.token_embedding.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
=======
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Four query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 3))
        )
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
=======
        self.token_emb = LowRankTokenEmbedding(cfg.vocab_size, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        # The learned low-rank vocabulary matrix is shared exactly between
        # token lookup and output classification.
        self.lm_head = LowRankTiedHead(self.token_emb)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor to preserve the qualified RNG
            # stream, then select the equivalent last-coordinate-zero gauge.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LowRankTiedHead):
            # This draw occurs at the same point as initialization of the
            # baseline tied head and becomes the effective shared matrix.
            token_embedding = module.token_embedding
            full = token_embedding.codes.new_empty(
                token_embedding.num_embeddings,
                token_embedding.embedding_dim,
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            token_embedding.initialize_from_full(full)
        elif isinstance(module, LowRankTokenEmbedding):
            # Preserve the baseline token-embedding initialization draw. As in
            # the tied baseline, the later head visit supplies the final draw.
            full = module.codes.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            module.initialize_from_full(full)
        elif isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor to preserve the qualified RNG
            # stream, then select the equivalent last-coordinate-zero gauge.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE