MECHANISM: Orthonormal positional common-mode gauge elimination

HYPOTHESIS: Representing the fourth-to-last positional row in an orthonormal seven-dimensional zero-mean basis will reduce the model from 1,628 to 1,627 parameters while retaining at least 99% accuracy, because it removes only the exact common-mode gauge while preserving the randomly initialized function and avoiding cross-position gradient interference.

INTENDED_EDIT: Compact the fourth-to-last positional row into seven orthonormal contrast coordinates, reconstruct its eight zero-mean coordinates during forward passes, and retain the two successful learned positional ties and final-row zero anchor.

EVIDENCE: Two learned cross-position gauge ties achieved 99.91%, but a third tie collapsed to 53.13% and blocking its reverse gradient fell to 27.69%; this motivates eliminating the same redundant scalar with an independent, orthonormal within-row gauge parameterization instead of another shared scalar.

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
    """Positional embedding with orthogonal, fixed, and dynamically tied gauges."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

        basis = self.weight.detach().new_zeros(embedding_dim, embedding_dim - 1)
        for col in range(embedding_dim - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("gauge_basis", basis, persistent=False)

        flat_weight = self.weight.detach().flatten()
        self.gauge_index = (num_embeddings - 3) * embedding_dim - 1
        tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1
        self.tie_index = tie_flat_index - 1
        self.anchor_index = anchor_flat_index - 2
        gauge_start = self.gauge_index - (embedding_dim - 1)
        gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[gauge_start : self.gauge_index + 1]
        )
        compact_weight = torch.cat(
            (
                flat_weight[:gauge_start],
                gauge_coords,
                flat_weight[self.gauge_index + 1 : tie_flat_index],
                flat_weight[tie_flat_index + 1 : anchor_flat_index],
                flat_weight[anchor_flat_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        gauge_start = self.gauge_index - (self.embedding_dim - 1)
        gauge_row = (
            self.gauge_basis
            @ self.weight[gauge_start : self.gauge_index]
        )
        flat_weight = torch.cat(
            (
                self.weight[:gauge_start],
                gauge_row,
                self.weight[self.gauge_index : self.tie_index],
                self.weight[1:2],
                self.weight[self.tie_index : self.anchor_index],
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
            flat_weight = full_weight.flatten()
            compact_weight = torch.cat(
                (
                    flat_weight[: module.tie_index],
                    flat_weight[module.tie_index + 1 : module.anchor_index],
                    flat_weight[module.anchor_index + 1 : -1],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
=======
        if isinstance(module, PositionAnchoredEmbedding):
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            flat_weight = full_weight.flatten()
            gauge_start = module.gauge_index - (module.embedding_dim - 1)
            tie_flat_index = module.tie_index + 1
            anchor_flat_index = module.anchor_index + 2
            gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[gauge_start : module.gauge_index + 1]
            )
            compact_weight = torch.cat(
                (
                    flat_weight[:gauge_start],
                    gauge_coords,
                    flat_weight[
                        module.gauge_index + 1 : tie_flat_index
                    ],
                    flat_weight[
                        tie_flat_index + 1 : anchor_flat_index
                    ],
                    flat_weight[anchor_flat_index + 1 : -1],
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
>>>>>>> REPLACE