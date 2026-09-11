MECHANISM: Second token/position translation-gauge quotient

HYPOTHESIS: Centering a second positional content coordinate across sequence positions will reduce the model from 1,546 to 1,545 parameters while retaining at least 99% accuracy, because its omitted position-common component is representable by the vocabulary-common component of the tied token embedding and is softmax-null at the output.

INTENDED_EDIT: Reparameterize the final two positional content coordinates with the existing mean-free position basis instead of only the final coordinate, leaving their position-varying capacity and all training settings unchanged.

EVIDENCE: The current 1,546-parameter model achieves 99.93% accuracy with one position-common mode already removed; extending that successful gauge quotient preserves positional variation, unlike the previously noted failures from pruning positional representation channels.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with one position-common mode removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 3))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        position_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            position_basis[: j + 1, j] = 1.0 / scale
            position_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("position_basis", position_basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        leading = F.embedding(idx, self.weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat((leading, last), dim=-1)
        return coordinates @ self.basis.transpose(0, 1)
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with two position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 4))
        self.last_coordinates = nn.Parameter(torch.empty(2, num_embeddings - 1))

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
        for j in range(embedding_dim - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("basis", basis, persistent=False)

        position_basis = torch.zeros(num_embeddings, num_embeddings - 1)
        for j in range(num_embeddings - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            position_basis[: j + 1, j] = 1.0 / scale
            position_basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("position_basis", position_basis, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        leading = F.embedding(idx, self.weight)
        last_weights = self.position_basis @ self.last_coordinates.transpose(0, 1)
        last = F.embedding(idx, last_weights)
        coordinates = torch.cat((leading, last), dim=-1)
        return coordinates @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove only one position-common mode.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, :-1])
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove two position-common modes.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, :-2])
                module.last_coordinates.copy_(
                    projected[:, -2:].transpose(0, 1) @ module.position_basis
                )
>>>>>>> REPLACE