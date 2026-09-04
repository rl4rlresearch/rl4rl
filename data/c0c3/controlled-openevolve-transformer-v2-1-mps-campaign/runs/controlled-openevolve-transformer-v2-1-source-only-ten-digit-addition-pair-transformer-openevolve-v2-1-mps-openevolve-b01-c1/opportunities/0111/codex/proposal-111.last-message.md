MECHANISM: First-coordinate positional common-mode quotient

HYPOTHESIS: Mean-centering positional content coordinate zero and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,533 to 1,532 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Add a third mean-free positional coordinate, reconstruct it from the position basis, and transfer its initialization mean into token embeddings so initial token-plus-position inputs remain unchanged.

EVIDENCE: The second-coordinate positional quotient reached 99.84%, while the inverse first-coordinate token-centering quotient reached 93.48% despite transferring its common component into decayed positional parameters; applying the successful transfer direction to this untested positional coordinate is the most targeted next reduction.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with two position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 4))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with three position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 5))
        self.first_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_first_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining = F.embedding(idx, self.weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
        second = F.embedding(idx, second_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (remaining[..., :1], second, remaining[..., 1:], last), dim=-1
        )
        return coordinates @ self.basis.transpose(0, 1)
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining = F.embedding(idx, self.weight)
        first_weight = (
            self.position_basis @ self.first_coordinate
        ).unsqueeze(1)
        first = F.embedding(idx, first_weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
        second = F.embedding(idx, second_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat((first, second, remaining, last), dim=-1)
        return coordinates @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second positional common mode. The corresponding token shift is also
        # softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
=======
        # Preserve the initialized token-plus-position inputs after removing the
        # first and second positional common modes. The corresponding token
        # shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb.removed_first_common
            )
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from the
            # second and final retained content coordinates.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(
                    torch.cat((projected[:, :1], projected[:, 2:-1]), dim=1)
                )
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from the
            # first, second, and final retained content coordinates.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, 2:-1])
                module.first_coordinate.copy_(
                    projected[:, 0] @ module.position_basis
                )
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.removed_second_common.copy_(projected[:, 1].mean())
>>>>>>> REPLACE