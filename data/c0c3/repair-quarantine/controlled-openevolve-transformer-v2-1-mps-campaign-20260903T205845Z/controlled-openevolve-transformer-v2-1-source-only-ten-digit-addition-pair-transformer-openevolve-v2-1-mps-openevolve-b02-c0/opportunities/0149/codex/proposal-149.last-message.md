MECHANISM: Per-position residual common-mode gauge elimination

HYPOTHESIS: Constraining positional row `max_seq_len - 12` to zero mean will reduce the model from 1,576 to 1,575 parameters while retaining at least 99% accuracy, because position-specific all-feature shifts are removed by every pre-LayerNorm and the final LayerNorm.

INTENDED_EDIT: Extend the positional embedding’s orthonormal zero-mean representation from eight consecutive rows to nine, preserving equivalent initialization in the reduced coordinates.

EVIDENCE: The 1,576-parameter model achieved 99.92% accuracy with eight positional rows already using this gauge; the two failed 1,575-parameter trials instead constrained LayerNorm or attention coordinates, motivating extension of the successful positional invariance.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
=======
        flat_weight = self.weight.detach().flatten()
        self.earliest_gauge_flat_start = (num_embeddings - 12) * embedding_dim
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
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
=======
                flat_weight[: self.earliest_gauge_flat_start],
                earliest_gauge_coords,
                earlier_gauge_coords,
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
=======
                self.weight[: self.earliest_gauge_index],
                earliest_gauge_row,
                earlier_gauge_row,
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
=======
                    flat_weight[: module.earliest_gauge_flat_start],
                    earliest_gauge_coords,
                    earlier_gauge_coords,
>>>>>>> REPLACE