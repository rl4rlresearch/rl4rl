MECHANISM: Incremental per-position residual common-mode gauge fixing

HYPOTHESIS: Constraining a ninth positional-embedding row to zero mean will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy, because its position-specific common-mode shift is removed by every LayerNorm without changing the learned computation.

INTENDED_EDIT: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization backward by one additional positional row, including compact reconstruction and initialization.

EVIDENCE: Extending the same positional gauge from seven to eight rows achieved 99.91% accuracy at 1,582 parameters, making a ninth row the closest controlled reduction.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
=======
        flat_weight = self.weight.detach().flatten()
        self.earliest_gauge_flat_start = (num_embeddings - 12) * embedding_dim
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.earlier_gauge_index = self.earlier_gauge_flat_start
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
=======
        self.earliest_gauge_index = self.earliest_gauge_flat_start
        self.earlier_gauge_index = self.earliest_gauge_index + embedding_dim - 1
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        earlier_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earlier_gauge_flat_start :
                self.earlier_gauge_flat_start + embedding_dim
            ]
        )
=======
        earliest_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earliest_gauge_flat_start :
                self.earliest_gauge_flat_start + embedding_dim
            ]
        )
        earlier_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earlier_gauge_flat_start :
                self.earlier_gauge_flat_start + embedding_dim
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                flat_weight[: self.earlier_gauge_flat_start],
                earlier_gauge_coords,
                preceding_gauge_coords,
=======
                flat_weight[: self.earliest_gauge_flat_start],
                earliest_gauge_coords,
                earlier_gauge_coords,
                preceding_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        earlier_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earlier_gauge_index : self.preceding_gauge_index]
        )
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        earliest_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earliest_gauge_index : self.earlier_gauge_index]
        )
        earlier_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earlier_gauge_index : self.preceding_gauge_index]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight[: self.earlier_gauge_index],
                earlier_gauge_row,
                preceding_gauge_row,
=======
                self.weight[: self.earliest_gauge_index],
                earliest_gauge_row,
                earlier_gauge_row,
                preceding_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            earlier_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earlier_gauge_flat_start :
                    module.earlier_gauge_flat_start + module.embedding_dim
                ]
            )
=======
            earliest_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earliest_gauge_flat_start :
                    module.earliest_gauge_flat_start + module.embedding_dim
                ]
            )
            earlier_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earlier_gauge_flat_start :
                    module.earlier_gauge_flat_start + module.embedding_dim
                ]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    flat_weight[: module.earlier_gauge_flat_start],
                    earlier_gauge_coords,
                    preceding_gauge_coords,
=======
                    flat_weight[: module.earliest_gauge_flat_start],
                    earliest_gauge_coords,
                    earlier_gauge_coords,
                    preceding_gauge_coords,
>>>>>>> REPLACE