MECHANISM: Learned low-rank vocabulary content subspace

HYPOTHESIS: Factorizing the token-content table through a learned six-dimensional subspace will reduce the model from 1,554 to 1,482 parameters while retaining at least 99% accuracy, because the 100 operand-pair tokens need learned behavioral similarity rather than seven independent content coordinates, while token-specific output biases and the full eight-dimensional transformer state remain available.

INTENDED_EDIT: Replace the independent seven-coordinate content vector for every vocabulary item with six learned token factors and a shared learned 6-by-7 mixing matrix, initialized by the best rank-six approximation of the original fresh random embedding; keep the independent mean-free token-row offsets, tied output projection, attention, positional representation, and training schedule unchanged.

EVIDENCE: The successful 1,554-parameter model still spends 798 parameters on an unconstrained per-token content table, while positional-coordinate removal failed despite preserving that table. This tests the different load-bearing assumption that every token requires seven unrelated content coordinates; learned factorization is plausible because operand-pair symbols can share latent behavior, and it preserves the row-offset channel required by the successful final-LayerNorm quotient.

<<<<<<< SEARCH
class MeanFreeTokenEmbedding(nn.Embedding):
    """Globally mean-free tied embedding with isolated token-row means."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()

        content_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            content_basis[: j + 1, j] = 1.0 / scale
            content_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("content_basis", content_basis, persistent=False)

        mean_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            mean_basis[: j + 1, j] = 1.0 / scale
            mean_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("mean_basis", mean_basis, persistent=False)

        row_average = full_weight.mean(dim=1)
        self.weight = nn.Parameter(
            ((full_weight - row_average.unsqueeze(1)) @ content_basis).clone()
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        centered = self.weight @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets
=======
class MeanFreeTokenEmbedding(nn.Embedding):
    """Low-rank tied token content with isolated token-row means."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        self.content_rank = embedding_dim - 2

        content_basis = torch.zeros(embedding_dim, embedding_dim - 1)
        for j in range(embedding_dim - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            content_basis[: j + 1, j] = 1.0 / scale
            content_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("content_basis", content_basis, persistent=False)

        mean_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            mean_basis[: j + 1, j] = 1.0 / scale
            mean_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("mean_basis", mean_basis, persistent=False)

        row_average = full_weight.mean(dim=1)
        content = (full_weight - row_average.unsqueeze(1)) @ content_basis
        u, s, vh = torch.linalg.svd(content, full_matrices=False)
        self.weight = nn.Parameter(
            (u[:, : self.content_rank] * s[: self.content_rank]).clone()
        )
        self.content_mix = nn.Parameter(vh[: self.content_rank].clone())
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        coordinates = self.weight @ self.content_mix
        centered = coordinates @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full @ module.content_basis)
                module.row_mean.zero_()
=======
        elif isinstance(module, MeanFreeTokenEmbedding):
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                content = full @ module.content_basis
                u, s, vh = torch.linalg.svd(content, full_matrices=False)
                module.weight.copy_(
                    u[:, : module.content_rank] * s[: module.content_rank]
                )
                module.content_mix.copy_(vh[: module.content_rank])
                module.row_mean.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
    embedding_params = [model.token_emb.weight, model.token_emb.row_mean]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.row_mean
    ]
=======
    embedding_params = [
        model.token_emb.weight,
        model.token_emb.content_mix,
        model.token_emb.row_mean,
    ]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.content_mix
        and param is not model.token_emb.row_mean
    ]
>>>>>>> REPLACE