MECHANISM: Complete positional common-mode quotient

HYPOTHESIS: Removing the position-common component of the sole remaining dense positional coordinate will reduce the model from 1,525 to 1,524 parameters while retaining at least 99% accuracy, because its constant input contribution can be transferred exactly to the matching token-embedding coordinate.

INTENDED_EDIT: Represent positional coordinate zero in the mean-free position basis, retain its removed common component in a buffer, and add that component to token embeddings at initialization.

EVIDENCE: Removing the common component of positional coordinate four preserved 99.88% accuracy at 1,525 parameters; this applies the same exact input-sum reparameterization to the only positional coordinate still storing a common mode.

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
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
=======
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
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining_weight = (self.position_basis @ self.weight).unsqueeze(1)
        remaining = F.embedding(idx, remaining_weight)
        second_weight = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second, third, and fourth positional common modes. The corresponding
        # token shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
=======
        # Preserve the initialized token-plus-position inputs after removing the
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
                projected = full @ module.basis
                module.weight.copy_(projected[:, :1])
                module.second_coordinate.copy_(
=======
                projected = full @ module.basis
                module.weight.copy_(
                    projected[:, 0] @ module.position_basis
                )
                module.second_coordinate.copy_(
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
=======
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_first_common.copy_(projected[:, 0].mean())
                module.removed_second_common.copy_(projected[:, 1].mean())
>>>>>>> REPLACE