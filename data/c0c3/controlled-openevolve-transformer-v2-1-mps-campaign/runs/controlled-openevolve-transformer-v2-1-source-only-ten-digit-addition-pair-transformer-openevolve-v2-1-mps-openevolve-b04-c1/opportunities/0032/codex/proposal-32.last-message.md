MECHANISM: Row-local positional shift quotient

HYPOTHESIS: Constraining every positional-embedding row to the seven-dimensional zero-mean subspace will reduce the model by `max_seq_len` parameters while retaining at least 99% accuracy, because each position has an independent all-ones residual shift erased by per-token LayerNorms.

INTENDED_EDIT: Replace the dense positional embedding with row-wise Householder coordinates that preserve the quotient initialization distribution without coupling optimization across positions.

EVIDENCE: The global positional-shift gauge failed at 49.92%, but it mixed coordinates across the entire positional table; the row-local orthonormal MLP-bias quotient achieved 99.92%, motivating a symmetry-aligned, independently parameterized quotient for each position.

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class RowShiftGaugeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.coordinates = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

        inv_sqrt = embedding_dim ** -0.5
        reflector = torch.full((embedding_dim,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coordinates, mean=0.0, std=0.02)

    def materialized_weight(self) -> torch.Tensor:
        weight = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.sum(
            weight * self.reflector, dim=-1, keepdim=True
        ) / self.reflector_norm_sq
        return weight - self.reflector * projection

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.materialized_weight())


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = ShiftGaugeEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = RowShiftGaugeEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
=======
        if isinstance(module, (ShiftGaugeEmbedding, RowShiftGaugeEmbedding)):
            module.reset_parameters()
>>>>>>> REPLACE