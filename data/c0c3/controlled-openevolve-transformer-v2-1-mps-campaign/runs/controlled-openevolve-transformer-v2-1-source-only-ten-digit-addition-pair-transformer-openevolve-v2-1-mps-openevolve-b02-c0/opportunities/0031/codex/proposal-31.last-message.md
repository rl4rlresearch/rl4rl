MECHANISM: Incremental positional-row common-mode gauge anchoring

HYPOTHESIS: Anchoring the final coordinate of the penultimate positional embedding row will reduce the model from 1,631 to 1,630 parameters while retaining at least 99% accuracy, because subtracting a row-specific all-coordinate offset is removed by every downstream LayerNorm and the same anchoring on the final row achieved 99.93%.

INTENDED_EDIT: Store both final positional rows without their last coordinate, reconstruct those coordinates as zero, and initialize each row by subtracting its sampled final coordinate.

EVIDENCE: Single-position embedding common-mode anchoring achieved 99.93% accuracy at 1,631 parameters, strongly motivating one incremental application of the identical gauge to the adjacent positional row.

<<<<<<< SEARCH
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with one common-mode coordinate anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight.detach().flatten()[:-1].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1)).view(
            self.num_embeddings, self.embedding_dim
        )
        return F.embedding(idx, weight)
=======
class PositionAnchoredEmbedding(nn.Embedding):
    """Positional embedding with two row-wise common-mode gauges anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        weight = torch.cat(
            (
                self.weight.detach()[:-2].flatten(),
                self.weight.detach()[-2:, :-1].flatten(),
            )
        )
        self.weight = nn.Parameter(weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        prefix_size = (self.num_embeddings - 2) * self.embedding_dim
        prefix = self.weight[:prefix_size].view(
            self.num_embeddings - 2, self.embedding_dim
        )
        tail = self.weight[prefix_size:].view(2, self.embedding_dim - 1)
        weight = torch.cat((prefix, F.pad(tail, (0, 1))), dim=0)
        return F.embedding(idx, weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            with torch.no_grad():
                module.weight.copy_(full_weight.flatten()[:-1])
=======
            anchors = full_weight[-2:, -1:].clone()
            full_weight[-2:].sub_(anchors)
            stored_weight = torch.cat(
                (full_weight[:-2].flatten(), full_weight[-2:, :-1].flatten())
            )
            with torch.no_grad():
                module.weight.copy_(stored_weight)
>>>>>>> REPLACE