MECHANISM: Initialization-anchored positional common mode

HYPOTHESIS: Freezing the fifth positional coordinate’s common component at its fresh-initialization value will reduce the model from 1,554 to 1,553 learned parameters while retaining at least 99% accuracy, because it preserves the successful model’s initial positional tensor exactly while removing only that component’s subsequent optimization.

INTENDED_EDIT: Split the fifth positional coordinate into a learned mean-free component and a persistent fixed common component captured from the original full-width initialization draw; retain the existing learned mean-free sixth coordinate and all other settings.

EVIDENCE: Setting the fifth positional common mode to zero collapsed accuracy to 38.22%, whereas retaining it produced 99.92%; anchoring it to its original random initialization distinguishes a required initial offset from a required learned degree of freedom without perturbing initialization.

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
    """Six-dimensional positions with one removed and one anchored common mode."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 4))
        self.penultimate_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
        self.register_buffer("penultimate_common", torch.empty(()))
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
        leading = F.embedding(idx, self.weight)
        penultimate_weight = (
            self.position_basis @ self.penultimate_coordinate
            + self.penultimate_common
        ).unsqueeze(1)
        penultimate = F.embedding(idx, penultimate_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat((leading, penultimate, last), dim=-1)
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
                module.weight.copy_(projected[:, :-2])
                penultimate = projected[:, -2]
                module.penultimate_coordinate.copy_(
                    penultimate @ module.position_basis
                )
                module.penultimate_common.copy_(penultimate.mean())
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
>>>>>>> REPLACE