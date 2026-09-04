MECHANISM: Vocabulary-translation gauge fixing

HYPOTHESIS: Centering tied token embeddings across vocabulary entries will reduce the verified 1,551-parameter model to 1,543 parameters while retaining at least 99% accuracy, because the removed shared embedding vector can be absorbed by the mean-zero positional embeddings up to an all-ones residual shift, while its output-head effect is only a class-independent logit shift.

INTENDED_EDIT: Parameterize the tied token embedding matrix in the vocabulary-mean-zero subspace and use its reconstructed full matrix for both input lookup and output logits.

EVIDENCE: The current mean-zero positional and residual-output design achieved 99.94% accuracy with 1,551 parameters, demonstrating that featurewise all-ones residual shifts are discarded by downstream normalization; vocabulary-wide translation of tied embeddings adds only a softmax-invariant common output-logit offset.

<<<<<<< SEARCH
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class MeanZeroLinear(nn.Module):
=======
class MeanZeroEmbedding(nn.Embedding):
    """Embedding parameterized on the subspace orthogonal to all-ones."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim - 1)
        self.register_buffer("basis", mean_zero_basis(embedding_dim), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return super().forward(idx) @ self.basis.transpose(0, 1)


class VocabCenteredEmbedding(nn.Embedding):
    """Tied embeddings with zero mean across vocabulary entries."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings - 1, embedding_dim)
        self.register_buffer("basis", mean_zero_basis(num_embeddings), persistent=False)

    def full_weight(self) -> torch.Tensor:
        return self.basis @ self.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.full_weight())


class MeanZeroLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE