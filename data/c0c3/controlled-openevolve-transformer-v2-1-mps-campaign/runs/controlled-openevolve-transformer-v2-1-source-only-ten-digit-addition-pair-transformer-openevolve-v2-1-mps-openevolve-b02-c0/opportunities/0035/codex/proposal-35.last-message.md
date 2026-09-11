MECHANISM: Dynamic cross-position gauge tying

HYPOTHESIS: Reusing the first learned positional scalar as the penultimate row’s final coordinate will reduce the model from 1,630 to 1,629 parameters while retaining at least 99% accuracy, because that coordinate is a pure positional common-mode gauge but remains randomly initialized and dynamically learned.

INTENDED_EDIT: Remove the penultimate positional row’s final scalar, reconstruct it from the first positional scalar, and preserve the successful final-row gauge initialization.

EVIDENCE: Fixing the second positional anchor locally reached 96.45%, while a learned LayerNorm tie reached 99.78% after its fixed counterpart reached only 37.17%; this motivates replacing the fixed second positional anchor with a learned tie.

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            with torch.no_grad():
                module.weight.copy_(full_weight.flatten()[:-1])
=======
        if isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            flat_weight = full_weight.flatten()
            compact_weight = torch.cat(
                (
                    flat_weight[: module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
>>>>>>> REPLACE