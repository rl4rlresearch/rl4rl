MECHANISM: Fifth positional common-mode quotient

HYPOTHESIS: Removing the position-common component of the fifth retained positional coordinate will reduce the model from 1,526 to 1,525 parameters while retaining at least 99% accuracy, because its input contribution can be transferred exactly to the corresponding token-embedding coordinate and remains softmax-null under the tied output projection.

INTENDED_EDIT: Represent positional coordinate four with a mean-free basis, transfer its initialized common component to token embeddings, and retain coordinate zero as the sole dense positional coordinate.

EVIDENCE: The verified 1,526-parameter design reached 99.82% accuracy while already quotienting four positional common modes by this mechanism; unlike the failed LayerNorm-scale reductions, this removes another instance of the same input-sum redundancy.

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
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with five position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 7))
        self.second_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.fifth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
=======
        self.register_buffer(
            "removed_fourth_common", torch.empty(()), persistent=False
        )
        self.register_buffer(
            "removed_fifth_common", torch.empty(()), persistent=False
        )

        basis = torch.zeros(embedding_dim, embedding_dim - 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        fifth_weight = (
            self.position_basis @ self.fifth_coordinate
        ).unsqueeze(1)
        fifth = F.embedding(idx, fifth_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (
                remaining,
                second,
                third,
                fourth,
                fifth,
                last,
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )

    @staticmethod
=======
            self.token_emb.weight[:, 3].add_(
                self.pos_emb.removed_fourth_common
            )
            self.token_emb.weight[:, 4].add_(
                self.pos_emb.removed_fifth_common
            )

    @staticmethod
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
=======
        elif isinstance(module, MeanFreePositionEmbedding):
            # Draw the original full-width initialization so later modules retain
            # the same RNG sequence, then remove position-common modes from all
            # retained content coordinates except the first.
            with torch.no_grad():
                full = torch.empty(
                    module.num_embeddings,
                    module.embedding_dim,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                projected = full @ module.basis
                module.weight.copy_(projected[:, :1])
                module.second_coordinate.copy_(
                    projected[:, 1] @ module.position_basis
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.fifth_coordinate.copy_(
                    projected[:, 4] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
                module.removed_second_common.copy_(projected[:, 1].mean())
                module.removed_third_common.copy_(projected[:, 2].mean())
                module.removed_fourth_common.copy_(projected[:, 3].mean())
                module.removed_fifth_common.copy_(projected[:, 4].mean())
>>>>>>> REPLACE