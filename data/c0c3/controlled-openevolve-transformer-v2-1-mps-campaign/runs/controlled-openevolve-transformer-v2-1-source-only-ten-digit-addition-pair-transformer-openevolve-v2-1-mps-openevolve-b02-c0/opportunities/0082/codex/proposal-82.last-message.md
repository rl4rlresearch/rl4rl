MECHANISM: Per-position residual common-mode gauge fixing

HYPOTHESIS: Constraining one additional positional-embedding row to zero mean will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because a position-specific common-mode shift is ignored by every LayerNorm and removed before logits by the final LayerNorm.

INTENDED_EDIT: Extend `PositionAnchoredEmbedding`’s orthogonal zero-mean parameterization from six positional rows to seven, including matching compact initialization and reconstruction.

EVIDENCE: The verified 1,584-parameter design achieved 99.97% accuracy with six positional rows already using this gauge, while extending `fc2` to a third constrained column collapsed accuracy to 73.46%; this tests an independent structural redundancy without further restricting `fc2`.

<<<<<<< SEARCH
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.leading_gauge_index = self.leading_gauge_flat_start
        self.zeroth_gauge_index = self.leading_gauge_index + embedding_dim - 1
        self.first_gauge_index = self.zeroth_gauge_index + embedding_dim - 1
        self.second_gauge_index = self.first_gauge_index + embedding_dim - 1
        self.third_gauge_index = self.second_gauge_index + embedding_dim - 1
        self.fourth_gauge_index = self.third_gauge_index + embedding_dim - 1
        self.gauge_end_index = self.fourth_gauge_index + embedding_dim - 1
        self.tie_index = self.gauge_end_index + embedding_dim - 1
        self.anchor_index = self.tie_index + embedding_dim - 1
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        leading_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.leading_gauge_flat_start :
                self.leading_gauge_flat_start + embedding_dim
            ]
        )
        zeroth_gauge_coords = (
=======
        preceding_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.preceding_gauge_flat_start :
                self.preceding_gauge_flat_start + embedding_dim
            ]
        )
        leading_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.leading_gauge_flat_start :
                self.leading_gauge_flat_start + embedding_dim
            ]
        )
        zeroth_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                flat_weight[: self.leading_gauge_flat_start],
                leading_gauge_coords,
                zeroth_gauge_coords,
=======
                flat_weight[: self.preceding_gauge_flat_start],
                preceding_gauge_coords,
                leading_gauge_coords,
                zeroth_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        leading_gauge_row = (
            self.gauge_basis
            @ self.weight[self.leading_gauge_index : self.zeroth_gauge_index]
        )
        zeroth_gauge_row = (
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        preceding_gauge_row = (
            self.gauge_basis
            @ self.weight[self.preceding_gauge_index : self.leading_gauge_index]
        )
        leading_gauge_row = (
            self.gauge_basis
            @ self.weight[self.leading_gauge_index : self.zeroth_gauge_index]
        )
        zeroth_gauge_row = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.weight[: self.leading_gauge_index],
                leading_gauge_row,
                zeroth_gauge_row,
=======
                self.weight[: self.preceding_gauge_index],
                preceding_gauge_row,
                leading_gauge_row,
                zeroth_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            leading_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.leading_gauge_flat_start :
                    module.leading_gauge_flat_start + module.embedding_dim
                ]
            )
            zeroth_gauge_coords = (
=======
            preceding_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.preceding_gauge_flat_start :
                    module.preceding_gauge_flat_start + module.embedding_dim
                ]
            )
            leading_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.leading_gauge_flat_start :
                    module.leading_gauge_flat_start + module.embedding_dim
                ]
            )
            zeroth_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
                    flat_weight[: module.leading_gauge_flat_start],
                    leading_gauge_coords,
                    zeroth_gauge_coords,
=======
                    flat_weight[: module.preceding_gauge_flat_start],
                    preceding_gauge_coords,
                    leading_gauge_coords,
                    zeroth_gauge_coords,
>>>>>>> REPLACE