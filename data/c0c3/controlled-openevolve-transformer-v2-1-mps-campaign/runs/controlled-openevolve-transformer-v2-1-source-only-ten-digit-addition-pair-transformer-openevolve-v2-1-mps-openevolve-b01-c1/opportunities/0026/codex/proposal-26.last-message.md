MECHANISM: Correctly deduplicated global tied-embedding shift quotient

HYPOTHESIS: Representing the tied token/output embedding with exactly one fewer learned coordinate will reduce the model from 1,586 to 1,585 parameters while retaining at least 99% accuracy, because the previous global-shift quotient achieved 99.99% accuracy and the removed direction is invisible to both LayerNorm and output softmax.

INTENDED_EDIT: Replace the tied full embedding parameter with an orthonormal mean-free flattened parameterization shared by input lookup and output projection, while preserving the original constructor and initialization RNG draws.

EVIDENCE: The prior tied-embedding quotient reached 99.99% accuracy but remained at 1,586 parameters, proving the constraint is accurate while showing that its previous implementation failed to remove an actual deduplicated parameter; the current key-bias-free design provides the 1,586-parameter baseline.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(idx, self.weight)
        return coordinates @ self.basis.transpose(0, 1)


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        coordinates = F.embedding(idx, self.weight)
        return coordinates @ self.basis.transpose(0, 1)


class MeanFreeTokenEmbedding(nn.Embedding):
    """Tied embedding modulo its globally constant, functionally null shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        size = num_embeddings * embedding_dim

        basis = torch.zeros(size, size - 1)
        for j in range(size - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        self.weight = nn.Parameter(
            (full_weight.reshape(-1) @ basis).clone()
        )

    def full_weight(self) -> torch.Tensor:
        return (self.basis @ self.weight).view(
            self.num_embeddings, self.embedding_dim
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class TiedMeanFreeOutput(nn.Linear):
    """Output projection sharing the quotient embedding's sole parameter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding: MeanFreeTokenEmbedding,
    ):
        # Preserve the constructor draw made by the original output Linear.
        super().__init__(in_features, out_features, bias=False)
        del self.weight
        object.__setattr__(self, "embedding", embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = MeanFreeTokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanFreePositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = TiedMeanFreeOutput(
            cfg.d_model, cfg.vocab_size, self.token_emb
        )

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, MeanFreePositionEmbedding):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TiedMeanFreeOutput):
            # The original tied output Linear initialized the shared embedding a
            # second time. Preserve that draw and retain its quotient coordinates.
            embedding = module.embedding
            with torch.no_grad():
                full = torch.empty(
                    embedding.num_embeddings,
                    embedding.embedding_dim,
                    device=embedding.weight.device,
                    dtype=embedding.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full.reshape(-1) @ module.basis)
        elif isinstance(module, MeanFreePositionEmbedding):
>>>>>>> REPLACE