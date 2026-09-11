MECHANISM: Gauge-fixed rank-seven lexical factorization

HYPOTHESIS: Eliminating the factorization’s 49-parameter internal change-of-basis redundancy will reduce the model from 1,509 to 1,460 parameters while retaining at least 99% accuracy, because the learned seven-dimensional token subspace is preserved through a seven-parameter graph over a fixed orthonormal basis.

INTENDED_EDIT: Replace the redundant learned 8×7 token projection with a fixed seven-dimensional orthonormal basis plus seven learned subspace-tilt parameters, retaining tied input embeddings and output logits.

EVIDENCE: The rank-seven design achieved 99.97% accuracy with 1,509 parameters, establishing that rank seven is sufficient. Unlike the failed token-code anchoring, this patch removes the exact invertible change-of-basis redundancy of a matrix factorization without constraining any token code or reducing its rank.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Learned low-rank token embedding shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if not 0 < rank < embedding_dim:
            raise ValueError("rank must be between zero and embedding_dim")

        self.code = nn.Embedding(num_embeddings, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(self.code(tokens))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.proj.weight.transpose(0, 1))
        return F.linear(latent, self.code.weight)
=======
class FactorizedTokenEmbedding(nn.Module):
    """Gauge-fixed low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        normal = torch.ones(embedding_dim) / math.sqrt(embedding_dim)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("normal", normal, persistent=False)

        # A rank-seven matrix has only seven subspace degrees of freedom
        # after its invertible latent change of basis is absorbed by code.
        self.tilt = nn.Parameter(torch.zeros(rank))

    def projection_weight(self) -> torch.Tensor:
        return self.basis + torch.outer(self.normal, self.tilt)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.projection_weight())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.projection_weight()
        latent = F.linear(x, weight.transpose(0, 1))
        return F.linear(latent, self.code.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.token_emb.proj.weight)
=======
        self.apply(self._init_weights)
>>>>>>> REPLACE