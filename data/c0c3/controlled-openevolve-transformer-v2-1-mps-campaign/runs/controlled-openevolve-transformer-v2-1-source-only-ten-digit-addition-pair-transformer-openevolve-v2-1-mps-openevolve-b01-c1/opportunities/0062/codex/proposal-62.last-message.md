MECHANISM: Head-local positional common-mode quotient

HYPOTHESIS: Constraining the third positional coordinate to be mean-free across positions will reduce the model from 1,554 to 1,553 parameters while retaining at least 99% accuracy, because this coordinate lies entirely within one four-channel attention head, unlike the fourth coordinate whose cross-head quotient reached only 92.63%.

INTENDED_EDIT: Reparameterize the third and sixth positional coordinates using `INPUT_LEN - 1` orthogonal position-axis coordinates, while preserving the original full-width initialization draw and all other model and training behavior.

EVIDENCE: Removing the sixth positional common mode achieved 99.92%, proving a second position-axis parameterization is viable; removing the fourth common mode reached 92.63%, substantially outperforming fifth-coordinate removal at 38.22% and motivating a test of the adjacent, head-local third coordinate.

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
    """Six-dimensional positions with two position-common modes removed."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 4))
        self.third_coordinate = nn.Parameter(torch.empty(num_embeddings - 1))
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
        retained = F.embedding(idx, self.weight)
        third_weight = (self.position_basis @ self.third_coordinate).unsqueeze(1)
        third = F.embedding(idx, third_weight)
        last_weight = (self.position_basis @ self.last_coordinate).unsqueeze(1)
        last = F.embedding(idx, last_weight)
        coordinates = torch.cat(
            (retained[..., :2], third, retained[..., 2:], last), dim=-1
        )
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
                module.weight.copy_(
                    torch.cat((projected[:, :2], projected[:, 3:-1]), dim=1)
                )
                module.third_coordinate.copy_(
                    projected[:, 2] @ module.position_basis
                )
                module.last_coordinate.copy_(
                    projected[:, -1] @ module.position_basis
                )
>>>>>>> REPLACE