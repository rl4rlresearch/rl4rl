MECHANISM: Nonadjacent orthonormal positional gauge elimination

HYPOTHESIS: Constraining positional row 1 to its seven-dimensional zero-mean subspace will reduce the model from 1,620 to 1,619 parameters while retaining at least 99% accuracy, indicating that the failed tenth-to-last-row extension was position-specific rather than a general limit of orthonormal gauge removal.

INTENDED_EDIT: Compact positional row 1 into independent orthonormal contrast coordinates while preserving the six successful late-row gauges, two learned ties, and final-row anchor.

EVIDENCE: Six consecutive orthonormal positional gauge reductions retained 99.82%–99.98% accuracy, including 99.93% at 1,620 parameters, while only the adjacent tenth-to-last row failed at 81.77%; testing a nonadjacent early row isolates positional sensitivity.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
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
=======
        flat_weight = self.weight.detach().flatten()
        self.early_gauge_flat_start = embedding_dim
        self.leading_gauge_flat_start = (num_embeddings - 9) * embedding_dim
        self.zeroth_gauge_flat_start = (num_embeddings - 8) * embedding_dim
        self.first_gauge_flat_start = (num_embeddings - 7) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 6) * embedding_dim
        self.third_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.fourth_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.early_gauge_index = self.early_gauge_flat_start
        self.early_gauge_end_index = self.early_gauge_index + embedding_dim - 1
        self.leading_gauge_index = self.leading_gauge_flat_start - 1
        self.zeroth_gauge_index = self.leading_gauge_index + embedding_dim - 1
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
        early_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.early_gauge_flat_start :
                self.early_gauge_flat_start + embedding_dim
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
        compact_weight = torch.cat(
            (
                flat_weight[: self.leading_gauge_flat_start],
                leading_gauge_coords,
=======
        compact_weight = torch.cat(
            (
                flat_weight[: self.early_gauge_flat_start],
                early_gauge_coords,
                flat_weight[
                    self.early_gauge_flat_start + embedding_dim :
                    self.leading_gauge_flat_start
                ],
                leading_gauge_coords,
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        leading_gauge_row = (
            self.gauge_basis
            @ self.weight[self.leading_gauge_index : self.zeroth_gauge_index]
        )
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        early_gauge_row = (
            self.gauge_basis
            @ self.weight[self.early_gauge_index : self.early_gauge_end_index]
        )
        leading_gauge_row = (
            self.gauge_basis
            @ self.weight[self.leading_gauge_index : self.zeroth_gauge_index]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        flat_weight = torch.cat(
            (
                self.weight[: self.leading_gauge_index],
                leading_gauge_row,
=======
        flat_weight = torch.cat(
            (
                self.weight[: self.early_gauge_index],
                early_gauge_row,
                self.weight[
                    self.early_gauge_end_index : self.leading_gauge_index
                ],
                leading_gauge_row,
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
            early_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.early_gauge_flat_start :
                    module.early_gauge_flat_start + module.embedding_dim
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
            compact_weight = torch.cat(
                (
                    flat_weight[: module.leading_gauge_flat_start],
                    leading_gauge_coords,
=======
            compact_weight = torch.cat(
                (
                    flat_weight[: module.early_gauge_flat_start],
                    early_gauge_coords,
                    flat_weight[
                        module.early_gauge_flat_start + module.embedding_dim :
                        module.leading_gauge_flat_start
                    ],
                    leading_gauge_coords,
>>>>>>> REPLACE