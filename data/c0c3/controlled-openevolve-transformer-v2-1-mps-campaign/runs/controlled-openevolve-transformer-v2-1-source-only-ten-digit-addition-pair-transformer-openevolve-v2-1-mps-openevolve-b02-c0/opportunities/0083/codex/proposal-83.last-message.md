MECHANISM: Per-position residual common-mode gauge fixing

HYPOTHESIS: Constraining an eighth positional-embedding row to zero mean will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy, because its position-specific common-mode shift is ignored by pre-LayerNorm sublayers and removed by the final LayerNorm.

INTENDED_EDIT: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization backward by one additional positional row, including matching compact initialization and reconstruction.

EVIDENCE: The immediately preceding identical extension from six to seven constrained positional rows achieved 99.83% accuracy at 1,583 parameters, making an eighth row the closest controlled reduction.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.preceding_gauge_index = self.preceding_gauge_flat_start
        self.leading_gauge_index = self.preceding_gauge_index + embedding_dim - 1
        self.zeroth_gauge_index = self.leading_gauge_index + embedding_dim - 1
        self.first_gauge_index = self.zeroth_gauge_index + embedding_dim - 1
        self.second_gauge_index = self.first_gauge_index + embedding_dim - 1
        self.third_gauge_index = self.second_gauge_index + embedding_dim - 1
        self.fourth_gauge_index = self.third_gauge_index + embedding_dim - 1
        self.gauge_end_index = self.fourth_gauge_index + embedding_dim - 1
        self.tie_index = self.gauge_end_index + embedding_dim - 1
        self.anchor_index = self.tie_index + embedding_dim - 1
=======
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.earlier_gauge_index = self.earlier_gauge_flat_start
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
        self.leading_gauge_index = self.preceding_gauge_index + embedding_dim - 1
        self.zeroth_gauge_index = self.leading_gauge_index + embedding_dim - 1
        self.first_gauge_index = self.zeroth_gauge_index + embedding_dim - 1
        self.second_gauge_index = self.first_gauge_index + embedding_dim - 1
        self.third_gauge_index = self.second_gauge_index + embedding_dim - 1
        self.fourth_gauge_index = self.third_gauge_index + embedding_dim - 1
        self.gauge_end_index = self.fourth_gauge_index + embedding_dim - 1
        self.tie_index = self.gauge_end_index + embedding_dim - 1
        self.anchor_index = self.tie_index + embedding_dim - 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        preceding_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.preceding_gauge_flat_start :
                self.preceding_gauge_flat_start + embedding_dim
            ]
        )
=======
        earlier_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earlier_gauge_flat_start :
                self.earlier_gauge_flat_start + embedding_dim
            ]
        )
        preceding_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.preceding_gauge_flat_start :
                self.preceding_gauge_flat_start + embedding_dim
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                flat_weight[: self.preceding_gauge_flat_start],
                preceding_gauge_coords,
                leading_gauge_coords,
=======
                flat_weight[: self.earlier_gauge_flat_start],
                earlier_gauge_coords,
                preceding_gauge_coords,
                leading_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        preceding_gauge_row = (
            self.gauge_basis
            @ self.weight[self.preceding_gauge_index : self.leading_gauge_index]
        )
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        earlier_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earlier_gauge_index : self.preceding_gauge_index]
        )
        preceding_gauge_row = (
            self.gauge_basis
            @ self.weight[self.preceding_gauge_index : self.leading_gauge_index]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight[: self.preceding_gauge_index],
                preceding_gauge_row,
                leading_gauge_row,
=======
                self.weight[: self.earlier_gauge_index],
                earlier_gauge_row,
                preceding_gauge_row,
                leading_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            preceding_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.preceding_gauge_flat_start :
                    module.preceding_gauge_flat_start + module.embedding_dim
                ]
            )
=======
            earlier_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earlier_gauge_flat_start :
                    module.earlier_gauge_flat_start + module.embedding_dim
                ]
            )
            preceding_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.preceding_gauge_flat_start :
                    module.preceding_gauge_flat_start + module.embedding_dim
                ]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    flat_weight[: module.preceding_gauge_flat_start],
                    preceding_gauge_coords,
                    leading_gauge_coords,
=======
                    flat_weight[: module.earlier_gauge_flat_start],
                    earlier_gauge_coords,
                    preceding_gauge_coords,
                    leading_gauge_coords,
>>>>>>> REPLACE