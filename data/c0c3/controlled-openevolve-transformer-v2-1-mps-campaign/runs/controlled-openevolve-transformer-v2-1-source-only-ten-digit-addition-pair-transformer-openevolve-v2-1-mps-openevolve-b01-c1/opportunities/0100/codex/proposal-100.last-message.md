MECHANISM: Learned rank-five tied token manifold

HYPOTHESIS: Replacing independent seven-dimensional token content vectors with a learned rank-five factorization will reduce the model from 1,535 to 1,360 parameters while retaining at least 99% accuracy, because token identities remain independently learnable and retain independent output offsets while the model learns a shared vocabulary geometry.

INTENDED_EDIT: Factorize the tied mean-free token embedding and output matrix into per-token rank-five coordinates and a learned map into the seven-dimensional centered model space; initialize it with the rank-five SVD of each fresh embedding draw and exempt both factors from weight decay.

EVIDENCE: The 1,535-parameter model reached 99.87%, while repeated removal of isolated MLP or attention coordinates caused highly brittle optimization failures. Those attempts assumed the dense per-token representation was load-bearing and targeted small downstream redundancies; this patch instead tests whether the dominant token-parameter block can learn addition through a substantially smaller shared latent vocabulary manifold.

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
    """Tied embedding learned through a rank-five mean-free token manifold."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        self.content_rank = embedding_dim - 3

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
        coordinates = (
            full_weight - row_average.unsqueeze(1)
        ) @ content_basis
        u, singular_values, vh = torch.linalg.svd(
            coordinates, full_matrices=False
        )
        self.weight = nn.Parameter(
            (u[:, : self.content_rank] * singular_values[: self.content_rank]).clone()
        )
        self.content_map = nn.Parameter(vh[: self.content_rank].clone())
        self.row_mean = nn.Parameter(
            (math.sqrt(embedding_dim) * (row_average @ mean_basis)).clone()
        )

    def initialize_content(self, full_weight: torch.Tensor) -> None:
        coordinates = full_weight @ self.content_basis
        u, singular_values, vh = torch.linalg.svd(
            coordinates, full_matrices=False
        )
        self.weight.copy_(
            u[:, : self.content_rank] * singular_values[: self.content_rank]
        )
        self.content_map.copy_(vh[: self.content_rank])

    def full_weight(self) -> torch.Tensor:
        centered = (
            self.weight @ self.content_map
        ) @ self.content_basis.transpose(0, 1)
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
                embedding.initialize_content(full)
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
                module.initialize_content(full)
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
        model.token_emb.content_map,
        model.token_emb.row_mean,
    ]
    decay_params = [
        param
        for param in model.parameters()
        if param is not model.token_emb.weight
        and param is not model.token_emb.content_map
        and param is not model.token_emb.row_mean
    ]
>>>>>>> REPLACE