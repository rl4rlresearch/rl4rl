MECHANISM: Second position-common-mode quotient with initialization transfer

HYPOTHESIS: Removing a second position-common mode will reduce the model from 1,534 to 1,533 parameters while retaining at least 99% accuracy, because the quotient preserves initial inputs exactly by transferring the removed common component into the tied token embedding.

INTENDED_EDIT: Mean-center the final two position-content coordinates instead of only the final coordinate, and transfer the newly removed initialization component into the matching non-decayed token coordinate.

EVIDENCE: The 1,534-parameter design achieves 99.91% with one position-common mode already removed. The failed inverse token-centering experiment moved a common component into decayed position parameters; this patch instead extends the successful positional constraint and moves its component into the token parameters already exempted from weight decay.

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
        self.last_coordinates = nn.Parameter(torch.empty(num_embeddings - 1, 2))
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
        last_weights = self.position_basis @ self.last_coordinates
        last = F.embedding(idx, last_weights)
        coordinates = torch.cat((leading, last), dim=-1)
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
                module.last_coordinates.copy_(
                    module.position_basis.transpose(0, 1) @ projected[:, -2:]
                )
                module._initial_common = projected[:, -2].mean().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Preserve the current design's initialized token-plus-position inputs:
        # the tied output changes only by a vocabulary-common logit offset.
        with torch.no_grad():
            self.token_emb.weight[:, self.pos_emb.weight.shape[1]].add_(
                self.pos_emb._initial_common
            )
        del self.pos_emb._initial_common

    @staticmethod
>>>>>>> REPLACE