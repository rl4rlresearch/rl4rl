MECHANISM: Fourth-coordinate positional common-mode quotient

HYPOTHESIS: Mean-centering positional content coordinate three and transferring its removed common component into the matching non-decayed token coordinate will reduce the model from 1,532 to 1,531 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Represent positional content coordinate three in the position-mean-free basis, reconstruct it during the forward pass, and transfer its initialization mean into token embeddings to preserve initial token-plus-position inputs.

EVIDENCE: Positional coordinate two achieved 100% accuracy after coordinate one achieved 99.84%, while coordinates zero and four failed; coordinate three is the closest untested positional mode to the two successful adjacent coordinates.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with three position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 5))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer(
            "removed_second_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_third_common", torch.empty(()), persistent=False
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_weight = (
            self.position_basis @ self.third_coordinate
        ).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (remaining[..., :1], second, third, remaining[..., 1:], last), dim=-1
        )
=======
        third_weight = (
            self.position_basis @ self.third_coordinate
        ).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (
                remaining[..., :1],
                second,
                third,
                fourth,
                remaining[..., 1:],
                last,
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve the initialized token-plus-position inputs after removing the
        # second and third positional common modes. The corresponding token
        # shifts are also softmax-null in the tied output projection.
        with torch.no_grad():
            self.token_emb.weight[:, 1].add_(
                self.pos_emb.removed_second_common
            )
            self.token_emb.weight[:, 2].add_(
                self.pos_emb.removed_third_common
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from the
            # second, third, and final retained content coordinates.
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
                    torch.cat((projected[:, :1], projected[:, 3:-1]), dim=1)
                )
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
=======
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
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
                module.removed_fourth_common.copy_(projected[:, 3].mean())
>>>>>>> REPLACE