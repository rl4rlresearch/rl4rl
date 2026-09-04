MECHANISM: Adjacent positional common-mode gauge fixing

HYPOTHESIS: Constraining the immediately preceding positional-embedding row to zero mean will reduce the model to 1,577 parameters while retaining at least 99% accuracy, because per-position common-mode shifts are removed by every LayerNorm and the initialization preserves the verified model modulo that exact invariance.

INTENDED_EDIT: Extend the existing eight-row orthogonal positional gauge to the adjacent ninth row, using the same full initialization draw and basis projection.

EVIDENCE: The verified 1,578-parameter design achieves 99.97% accuracy while already constraining eight adjacent positional rows with this exact gauge; this repeats that established construction instead of extending the fourth `ln1` tie that fell to 52.63%.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
=======
        flat_weight = self.weight.detach().flatten()
        self.prior_gauge_flat_start = (num_embeddings - 12) * embedding_dim
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.earlier_gauge_index = self.earlier_gauge_flat_start
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
=======
        self.prior_gauge_index = self.prior_gauge_flat_start
        self.earlier_gauge_index = self.prior_gauge_index + embedding_dim - 1
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
        prior_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.prior_gauge_flat_start :
                self.prior_gauge_flat_start + embedding_dim
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
                flat_weight[: self.prior_gauge_flat_start],
                prior_gauge_coords,
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
        prior_gauge_row = (
            self.gauge_basis
            @ self.weight[self.prior_gauge_index : self.earlier_gauge_index]
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
                self.weight[: self.prior_gauge_index],
                prior_gauge_row,
                earlier_gauge_row,
                preceding_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            flat_weight = full_weight.flatten()
            earlier_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.earlier_gauge_flat_start :
                    module.earlier_gauge_flat_start + module.embedding_dim
                ]
            )
=======
            flat_weight = full_weight.flatten()
            prior_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.prior_gauge_flat_start :
                    module.prior_gauge_flat_start + module.embedding_dim
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
                    flat_weight[: module.prior_gauge_flat_start],
                    prior_gauge_coords,
                    earlier_gauge_coords,
                    preceding_gauge_coords,
>>>>>>> REPLACE