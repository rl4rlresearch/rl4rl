MECHANISM: Alternative-axis positional common-mode quotient

HYPOTHESIS: Constraining the fourth positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because it preserves all six position-dependent coordinates and leaves the essential fifth-coordinate common component learned.

INTENDED_EDIT: Reparameterize the fourth and sixth positional coordinates with `INPUT_LEN - 1` orthogonal position-axis coordinates, preserving the full-width initialization draw and all other model and training settings.

EVIDENCE: Removing the sixth coordinate’s common mode achieved 99.92%, while removing the fifth coordinate’s common mode fell to 38.22%; testing the fourth coordinate determines whether that failure is specific to the fifth coordinate rather than a general limit on removing a second positional common mode.

<<<<<<< SEARCH
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with one position-common mode removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 3))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
=======
class MeanFreePositionEmbedding(nn.Module):
    """Six-dimensional positions with two selected common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 4))
        self.fourth_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.last_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        leading = F.embedding(idx, self.weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat((leading, last), dim=-1)
        return coordinates @ self.basis.transpose(0, 1)
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        free = F.embedding(idx, self.weight)
        fourth_weight = (
            self.position_basis @ self.fourth_coordinate
        ).unsqueeze(1)
        fourth = F.embedding(idx, fourth_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (free[..., :3], fourth, free[..., 3:], last), dim=-1
        )
        return coordinates @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                projected = full @ module.basis
                module.weight.copy_(projected[:, :-1])
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
=======
                projected = full @ module.basis
                module.weight.copy_(
                    torch.cat((projected[:, :3], projected[:, 4:-1]), dim=1)
                )
                module.fourth_coordinate.copy_(
                    projected[:, 3] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
>>>>>>> REPLACE