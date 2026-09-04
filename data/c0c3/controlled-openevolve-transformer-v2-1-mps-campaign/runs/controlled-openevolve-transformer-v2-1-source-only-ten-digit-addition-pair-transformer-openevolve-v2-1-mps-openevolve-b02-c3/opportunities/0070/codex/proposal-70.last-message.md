MECHANISM: Hybrid learned–sinusoidal lexical rank

HYPOTHESIS: A 717-parameter transformer will retain at least 99% accuracy because rank five is preserved while only four lexical coordinates require independent per-token learning; a generic fixed fifth coordinate with learned global amplitude can supply the missing classifier-visible direction.

INTENDED_EDIT: Replace each five-coordinate learned token code with four learned coordinates plus one normalized sinusoidal token-identity coordinate, retaining the existing five-dimensional projection, attention computation, MLP, and terminal calibration.

EVIDENCE: The 829-parameter rank-five design achieved 99.88%, while the rank-four bottleneck failed; this establishes that a fifth lexical direction is load-bearing but does not establish that all 110 values along it need independent parameters. The current 826-parameter design’s 99.98% provides margin for testing this alternative full-rank representation.

<<<<<<< SEARCH
class FactorizedTokenEmbedding(nn.Module):
    """Fixed-subspace low-rank token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")

        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.code(tokens), self.basis)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(latent, self.code.weight)
=======
class FactorizedTokenEmbedding(nn.Module):
    """Full-rank token map with one shared sinusoidal lexical coordinate."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 2 or rank >= embedding_dim:
            raise ValueError("rank must be between two and embedding_dim - 1")

        self.rank = rank
        self.code = nn.Embedding(num_embeddings, rank - 1)
        self.fixed_scale = nn.Parameter(torch.tensor(0.02))

        fixed_code = torch.sin(
            torch.arange(num_embeddings, dtype=torch.float32) * math.sqrt(2.0)
        )
        fixed_code = fixed_code - fixed_code.mean()
        fixed_code = fixed_code / fixed_code.std(unbiased=False)
        self.register_buffer(
            "fixed_code", fixed_code.unsqueeze(1), persistent=False
        )

        basis = torch.zeros(embedding_dim, rank)
        for col in range(rank):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def lexical_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.code.weight, self.fixed_scale * self.fixed_code), dim=1
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.lexical_weight()[tokens], self.basis)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(latent, self.lexical_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
=======
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.rank + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.rank - 2)
        )
>>>>>>> REPLACE