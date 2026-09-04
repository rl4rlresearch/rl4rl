MECHANISM: Incremental dynamic cross-position gauge tying

HYPOTHESIS: Reusing the second learned positional scalar as the third-to-last row’s final coordinate will reduce the model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because it extends the successful randomly initialized learned positional tie to one adjacent row.

INTENDED_EDIT: Remove the third-to-last positional row’s final scalar, reconstruct it from the second positional scalar, and retain the existing penultimate learned tie and final-row zero anchor.

EVIDENCE: Replacing the penultimate row’s fixed-zero coordinate, which achieved only 96.45%, with a learned cross-position tie achieved 99.91% at 1,629 parameters; this directly motivates one incremental tie using a distinct learned scalar.

<<<<<<< SEARCH
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with fixed and dynamically tied gauge coordinates."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        flat_weight = self.weight.detach().flatten()
        self.anchor_index = (num_embeddings - 1) * embedding_dim - 1
        compact_weight = torch.cat(
            (
                flat_weight[: self.anchor_index],
                flat_weight[self.anchor_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.weight[: self.anchor_index],
                self.weight[:1],
                self.weight[self.anchor_index :],
                self.weight.new_zeros(1),
            )
        )
        weight = flat_weight.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(idx, weight)
=======
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with fixed and dynamically tied gauge coordinates."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        flat_weight = self.weight.detach().flatten()
        self.tie_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_index = (num_embeddings - 1) * embedding_dim - 1
        compact_weight = torch.cat(
            (
                flat_weight[: self.tie_index],
                flat_weight[self.tie_index + 1 : self.anchor_index],
                flat_weight[self.anchor_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.weight[: self.tie_index],
                self.weight[1:2],
                self.weight[self.tie_index : self.anchor_index - 1],
                self.weight[:1],
                self.weight[self.anchor_index - 1 :],
                self.weight.new_zeros(1),
            )
        )
        weight = flat_weight.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_weight = torch.cat(
                (
                    flat_weight[: module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
=======
            compact_weight = torch.cat(
                (
                    flat_weight[: module.tie_index],
                    flat_weight[module.tie_index + 1 : module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
>>>>>>> REPLACE