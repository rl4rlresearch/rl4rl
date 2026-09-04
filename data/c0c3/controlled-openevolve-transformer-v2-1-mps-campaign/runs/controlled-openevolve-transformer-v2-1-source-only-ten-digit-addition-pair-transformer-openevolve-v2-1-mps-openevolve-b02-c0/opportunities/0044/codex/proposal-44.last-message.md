MECHANISM: Independent orthonormal positional gauge extension

HYPOTHESIS: Constraining the fifth-to-last positional row to the seven-dimensional zero-mean subspace will reduce the model from 1,625 to 1,624 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Compact two adjacent positional rows into independent orthonormal contrast coordinates, preserving the existing learned ties and final-row anchor.

EVIDENCE: The first orthonormal positional gauge achieved 99.82% at 1,627 parameters, whereas adding a third cross-position tie collapsed to 53.13%; an independent orthonormal gauge extends the successful parameterization without cross-position gradient coupling.

<<<<<<< SEARCH
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
=======
        flat_weight = self.weight.detach().flatten()
        self.first_gauge_flat_start = (num_embeddings - 5) * embedding_dim
        self.second_gauge_flat_start = (num_embeddings - 4) * embedding_dim
        self.tie_flat_index = (num_embeddings - 2) * embedding_dim - 1
        self.anchor_flat_index = (num_embeddings - 1) * embedding_dim - 1

        self.first_gauge_index = self.first_gauge_flat_start
        self.second_gauge_index = self.first_gauge_index + embedding_dim - 1
        self.gauge_end_index = self.second_gauge_index + embedding_dim - 1
        self.tie_index = self.gauge_end_index + embedding_dim - 1
        self.anchor_index = self.tie_index + embedding_dim - 1

        first_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.first_gauge_flat_start :
                self.first_gauge_flat_start + embedding_dim
            ]
        )
        second_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.second_gauge_flat_start :
                self.second_gauge_flat_start + embedding_dim
            ]
        )
        compact_weight = torch.cat(
            (
                flat_weight[: self.first_gauge_flat_start],
                first_gauge_coords,
                second_gauge_coords,
                flat_weight[
                    self.second_gauge_flat_start + embedding_dim :
                    self.tie_flat_index
                ],
                flat_weight[
                    self.tie_flat_index + 1 : self.anchor_flat_index
                ],
                flat_weight[self.anchor_flat_index + 1 : -1],
            )
        )
        self.weight = nn.Parameter(compact_weight.clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        first_gauge_row = (
            self.gauge_basis
            @ self.weight[self.first_gauge_index : self.second_gauge_index]
        )
        second_gauge_row = (
            self.gauge_basis
            @ self.weight[self.second_gauge_index : self.gauge_end_index]
        )
        flat_weight = torch.cat(
            (
                self.weight[: self.first_gauge_index],
                first_gauge_row,
                second_gauge_row,
                self.weight[self.gauge_end_index : self.tie_index],
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
=======
            flat_weight = full_weight.flatten()
            first_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.first_gauge_flat_start :
                    module.first_gauge_flat_start + module.embedding_dim
                ]
            )
            second_gauge_coords = (
                module.gauge_basis.transpose(0, 1)
                @ flat_weight[
                    module.second_gauge_flat_start :
                    module.second_gauge_flat_start + module.embedding_dim
                ]
            )
            compact_weight = torch.cat(
                (
                    flat_weight[: module.first_gauge_flat_start],
                    first_gauge_coords,
                    second_gauge_coords,
                    flat_weight[
                        module.second_gauge_flat_start + module.embedding_dim :
                        module.tie_flat_index
                    ],
                    flat_weight[
                        module.tie_flat_index + 1 : module.anchor_flat_index
                    ],
                    flat_weight[module.anchor_flat_index + 1 : -1],
                )
            )
>>>>>>> REPLACE