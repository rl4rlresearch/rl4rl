MECHANISM: Stop-gradient cross-position gauge tying

HYPOTHESIS: Reusing the third learned positional scalar for the fourth-to-last row while blocking that row’s reverse gradient will reduce the model to 1,627 parameters and retain at least 99% accuracy by avoiding the cross-position gradient interference implicated by the prior 53.13% result.

INTENDED_EDIT: Remove the fourth-to-last positional row’s final scalar, reconstruct it from a detached view of the third learned positional scalar, and retain the two successful learned ties and final-row zero anchor.

EVIDENCE: Two dynamic positional ties retained 99.91% accuracy at 1,629 and 1,628 parameters, but adding a third tie collapsed to 53.13%; isolating the new tie’s backward coupling tests whether optimization interference, rather than the gauge reduction itself, caused that failure.

<<<<<<< SEARCH
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
=======
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with fixed and dynamically tied gauge coordinates."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        flat_weight = self.weight.detach().flatten()
        self.detached_tie_index = (num_embeddings - 3) * embedding_dim - 1
        self.tie_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_index = (num_embeddings - 1) * embedding_dim - 1
        compact_weight = torch.cat(
            (
                flat_weight[: self.detached_tie_index],
                flat_weight[self.detached_tie_index + 1 : self.tie_index],
                flat_weight[self.tie_index + 1 : self.anchor_index],
                flat_weight[self.anchor_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        flat_weight = torch.cat(
            (
                self.weight[: self.detached_tie_index],
                self.weight[2:3].detach(),
                self.weight[self.detached_tie_index : self.tie_index - 1],
                self.weight[1:2],
                self.weight[self.tie_index - 1 : self.anchor_index - 2],
                self.weight[:1],
                self.weight[self.anchor_index - 2 :],
                self.weight.new_zeros(1),
            )
        )
        weight = flat_weight.view(self.num_embeddings, self.embedding_dim)
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_weight = torch.cat(
                (
                    flat_weight[: module.tie_index],
                    flat_weight[module.tie_index + 1 : module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
=======
            compact_weight = torch.cat(
                (
                    flat_weight[: module.detached_tie_index],
                    flat_weight[
                        module.detached_tie_index + 1 : module.tie_index
                    ],
                    flat_weight[module.tie_index + 1 : module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
>>>>>>> REPLACE