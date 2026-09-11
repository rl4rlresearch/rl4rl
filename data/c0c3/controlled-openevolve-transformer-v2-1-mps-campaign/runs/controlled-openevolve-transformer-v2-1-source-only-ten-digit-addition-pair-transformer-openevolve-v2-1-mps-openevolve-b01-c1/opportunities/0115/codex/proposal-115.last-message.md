MECHANISM: Balanced residual positional common-mode quotient

HYPOTHESIS: Removing the equal-weight common mode of positional content coordinates zero and four will reduce the model from 1,531 to 1,530 parameters while retaining at least 99% accuracy, because it preserves a trainable orthogonal common mode and exactly preserves initialized token-plus-position inputs.

INTENDED_EDIT: Reparameterize the two remaining positional coordinates into mean-free variation plus one shared common component, transferring the removed balanced component into the corresponding non-decayed token coordinates.

EVIDENCE: Individual removal of coordinate zero reached 62.71% and coordinate four reached 69.82%, while coordinates one through three successfully supported the same quotient; a balanced constraint tests whether the two resistant coordinates need a common degree of freedom without fixing either coordinate outright.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with four position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 6))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with five position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings - 1, embedding_dim - 6)
        )
        self.remaining_common = nn.Parameter(torch.empty(()))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "remaining_common_direction",
            torch.tensor([1.0, -1.0]) / math.sqrt(2.0),
            persistent=False,
        )
        self.register_buffer(
            "removed_remaining_direction",
            torch.tensor([1.0, 1.0]) / math.sqrt(2.0),
            persistent=False,
        )
        self.register_buffer(
            "removed_remaining_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining = F.embedding(idx, self.weight)
        second_weight = (
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        remaining_weight = self.position_basis @ self.weight
        remaining_weight = remaining_weight + (
            self.remaining_common * self.remaining_common_direction
        )
        remaining = F.embedding(idx, remaining_weight)
        second_weight = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second, third, and fourth positional common modes. The corresponding
        # token shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_third_common
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )
=======
        # Preserve initialized token-plus-position inputs after removing the
        # balanced remaining mode and the second through fourth positional
        # common modes. These vocabulary-common token shifts are softmax-null.
        with torch.no_grad():
            removed_remaining = self.pos_emb.removed_remaining_common
            removed_direction = self.pos_emb.removed_remaining_direction
            self.token_emb.weight[:, 0].add_(
                removed_remaining * removed_direction[0]
            )
            self.token_emb.weight[:, 4].add_(
                removed_remaining * removed_direction[1]
            )
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_third_common
            )
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from the
            # second, third, fourth, and final retained content coordinates.
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
                    torch.cat((projected[:, :1], projected[:, 4:-1]), dim=1)
                )
                module.second_coordinate.copy_(
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove a balanced common mode from the
            # remaining pair and the common modes of coordinates one, two, three,
            # and five.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                remaining = torch.stack(
                    (projected[:, 0], projected[:, 4]), dim=1
                )
                module.weight.copy_(
                    module.position_basis.transpose(0, 1) @ remaining
                )
                remaining_mean = remaining.mean(dim=0)
                module.remaining_common.copy_(
                    remaining_mean @ module.remaining_common_direction
                )
                module.removed_remaining_common.copy_(
                    remaining_mean @ module.removed_remaining_direction
                )
                module.second_coordinate.copy_(
>>>>>>> REPLACE