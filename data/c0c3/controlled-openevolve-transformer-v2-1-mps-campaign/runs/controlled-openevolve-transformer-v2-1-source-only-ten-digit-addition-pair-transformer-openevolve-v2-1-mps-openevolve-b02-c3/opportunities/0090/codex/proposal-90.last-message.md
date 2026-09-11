MECHANISM: Quadratically lifted rank-four tied lexical manifold

HYPOTHESIS: A `803 - VOCAB_SIZE`-parameter transformer will retain at least 99% accuracy because a parameter-free quadratic lift restores five-dimensional token and classifier geometry from four learned coordinates, avoiding the linear-rank bottleneck that hurt the prior lexical reduction.

INTENDED_EDIT: Reconstruct the qualified 803-parameter bias constraints, then replace each five-coordinate token code with four learned coordinates whose pairwise products supply a fifth nonlinear feature shared by input embeddings and output logits.

EVIDENCE: The linear lexical-rank reduction still reached 95.12%, suggesting compact token codes retain most of the task but losing a classifier direction is load-bearing. The successful multiplicatively gated MLP shows learned low-dimensional coordinates can recover useful capacity through parameter-free products.

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
    """Quadratically lifted token map shared with the output classifier."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        if rank < 2 or rank >= embedding_dim - 1 or rank % 2:
            raise ValueError(
                "rank must be even and between two and embedding_dim - 2"
            )

        self.rank = rank
        self.lifted_dim = rank + 1
        self.code = nn.Embedding(num_embeddings, rank)

        basis = torch.zeros(embedding_dim, self.lifted_dim)
        for col in range(self.lifted_dim):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

    def lift(self, code: torch.Tensor) -> torch.Tensor:
        pairs = code.reshape(
            *code.shape[:-1], self.rank // 2, 2
        )
        interaction = pairs.prod(dim=-1).sum(dim=-1, keepdim=True)
        interaction = interaction / (
            math.sqrt(self.rank / 2) * 0.02
        )
        return torch.cat((code, interaction), dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.lift(self.code(tokens)), self.basis)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        latent = F.linear(x, self.basis.transpose(0, 1))
        return F.linear(latent, self.lift(self.code.weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 2
        )
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
            cfg.d_model, self.token_emb.lifted_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.lifted_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_bias = F.pad(self.final_bias, (0, 2))
=======
        final_bias = F.pad(self.final_bias, (0, 3))
>>>>>>> REPLACE