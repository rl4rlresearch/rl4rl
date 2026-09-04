MECHANISM: Final positional common-mode quotient

HYPOTHESIS: Removing the position-common component of the sole dense positional coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its input contribution transfers exactly to the corresponding token-embedding coordinate and adds only a vocabulary-common offset under the tied output projection.

INTENDED_EDIT: Represent positional coordinate zero in the mean-free position basis, preserve its initialized common component in token embeddings, and leave no dense positional coordinate.

EVIDENCE: Removing the same common mode from positional coordinate four achieved 99.88% accuracy at 1,525 parameters; this applies that already-successful exact quotient to the remaining dense coordinate.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with five position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 7))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with all position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_first_common", torch.empty(()), persistent=False
        )
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining = F.embedding(idx, self.weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first_weight = (self.position_basis @ self.weight).unsqueeze(1)
        remaining = F.embedding(idx, first_weight)
        second_weight = (
            self.position_basis @ self.second_coordinate
        ).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second, third, and fourth positional common modes. The corresponding
        # token shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
=======
        # Preserve the initialized token-plus-position inputs after removing
        # positional common modes. The corresponding token shifts are also
        # softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 0].add_(
                self.pos_emb.removed_first_common
            )
            self.token_emb.weight[:, 1].add_(
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from all
            # retained content coordinates except the first.
=======
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from all
            # retained content coordinates.
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight.copy_(projected[:, :1])
                module.second_coordinate.copy_(
=======
                module.weight.copy_(projected[:, 0] @ module.position_basis)
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.second_coordinate.copy_(
>>>>>>> REPLACE