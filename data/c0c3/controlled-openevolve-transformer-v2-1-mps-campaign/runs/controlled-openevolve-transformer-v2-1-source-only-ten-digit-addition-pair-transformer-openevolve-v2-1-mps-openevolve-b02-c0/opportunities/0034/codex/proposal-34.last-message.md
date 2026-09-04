MECHANISM: Locally initialized second positional-row gauge anchor

HYPOTHESIS: Fixing the penultimate positional row’s final coordinate at zero without shifting its other coordinates will reduce the model from 1,630 to 1,629 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Compact the final two positional rows by one scalar each, preserve the successful final-row subtraction initialization, and independently initialize the penultimate row’s surviving coordinates.

EVIDENCE: Subtracting an anchor from a second positional row failed at 72.89%, while replacing the failed globally shifted token anchor with a locally initialized fixed scalar recovered from 84.68% to 99.91%; this directly motivates testing the same localized initialization remedy for the second positional gauge.

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
    """Positional embedding with two common-mode coordinates anchored."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            torch.cat(
                (
                    full_weight[:-2].flatten(),
                    full_weight[-2, :-1],
                    full_weight[-1, :-1],
                )
            ).clone()
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        prefix_size = (self.num_embeddings - 2) * self.embedding_dim
        prefix = self.weight[:prefix_size].view(
            self.num_embeddings - 2, self.embedding_dim
        )
        tail = F.pad(
            self.weight[prefix_size:].view(2, self.embedding_dim - 1),
            (0, 1),
        )
        return F.embedding(idx, torch.cat((prefix, tail), dim=0))
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
            compact_weight = torch.cat(
                (
                    full_weight[:-2].flatten(),
                    full_weight[-2, :-1],
                    full_weight[-1, :-1],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
>>>>>>> REPLACE