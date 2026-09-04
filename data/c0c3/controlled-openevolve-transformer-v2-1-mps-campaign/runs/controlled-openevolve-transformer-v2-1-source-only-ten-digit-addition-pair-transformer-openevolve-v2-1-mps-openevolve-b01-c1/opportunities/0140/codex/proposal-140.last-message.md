MECHANISM: Adaptive low-rank tied token representation

HYPOTHESIS: Constraining the token-content table to a learned six-dimensional subspace will reduce the verified model from 1,520 to 1,448 parameters while retaining at least 99% accuracy, because the shared projection can learn which single token-feature direction to discard while preserving all six learned positional profiles.

INTENDED_EDIT: Factor the mean-free token-content embedding into per-token six-dimensional codes and a learned shared 6-by-7 projection, use the reconstructed embeddings for both input and tied output logits, and exempt both factors from weight decay.

EVIDENCE: Replacing positional lookup channels reduced accuracy to 98.89% or 74.41%, showing that positional flexibility is load-bearing; this instead preserves the verified attention, MLP, and positional computation while testing the unchallenged assumption that every token requires seven independent mean-free content coordinates.

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
    """Low-rank mean-free tied embedding with isolated token-row means."""

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
        content_coordinates = (
            full_weight - row_average.unsqueeze(1)
        ) @ content_basis
        self.weight = nn.Parameter(
            content_coordinates[:, : self.content_rank].clone()
        )
        self.content_mix = nn.Parameter(
            torch.eye(
                self.content_rank,
                embedding_dim - 1,
                device=full_weight.device,
                dtype=full_weight.dtype,
            )
        )
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def full_weight(self) -> torch.Tensor:
        centered_coordinates = self.weight @ self.content_mix
        centered = centered_coordinates @ self.content_basis.transpose(0, 1)
        row_offsets = (
            (self.mean_basis @ self.row_mean).unsqueeze(1)
            / math.sqrt(self.embedding_dim)
        )
        return centered + row_offsets
>>>>>>> REPLACE

<<<<<<< SEARCH
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full @ embedding.content_basis)
                embedding.row_mean.zero_()
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
                full.sub_(full.mean(dim=1, keepdim=True))
                coordinates = full @ embedding.content_basis
                embedding.content_mix.copy_(
                    torch.eye(
                        embedding.content_rank,
                        embedding.embedding_dim - 1,
                        device=embedding.weight.device,
                        dtype=embedding.weight.dtype,
                    )
                )
                embedding.weight.copy_(
                    coordinates[:, : embedding.content_rank]
                )
                embedding.row_mean.zero_()
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
                coordinates = full @ module.content_basis
                module.content_mix.copy_(
                    torch.eye(
                        module.content_rank,
                        module.embedding_dim - 1,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                    )
                )
                module.weight.copy_(
                    coordinates[:, : module.content_rank]
                )
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