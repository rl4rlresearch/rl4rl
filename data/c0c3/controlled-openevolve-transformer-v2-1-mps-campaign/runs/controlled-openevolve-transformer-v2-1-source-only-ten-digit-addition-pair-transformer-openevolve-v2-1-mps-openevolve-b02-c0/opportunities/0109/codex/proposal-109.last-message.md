MECHANISM: Last-coordinate positional common-shift gauge

HYPOTHESIS: Anchoring the adjacent positional row with coordinate differences will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because initialization changes that row only by an exactly LayerNorm-invisible common shift while avoiding the failed orthogonal Adam reparameterization.

INTENDED_EDIT: Store seven differences for positional row `num_embeddings - 12`, reconstruct its final coordinate as zero, and preserve full-size initialization draws and all existing gauges.

EVIDENCE: The orthogonal ninth-row gauge failed at 16.06%, but the verified 1,577-parameter model already contains a successful last-coordinate positional anchor; the analogous difference-coordinate gauge in `NormalizedInputLinear` also retained 99.93%.

<<<<<<< SEARCH
        flat_weight = self.weight.detach().flatten()
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
=======
        flat_weight = self.weight.detach().flatten()
        self.fixed_gauge_flat_start = (num_embeddings - 12) * embedding_dim
        self.earlier_gauge_flat_start = (num_embeddings - 11) * embedding_dim
        self.preceding_gauge_flat_start = (num_embeddings - 10) * embedding_dim
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.earlier_gauge_index = self.earlier_gauge_flat_start
        self.preceding_gauge_index = self.earlier_gauge_index + embedding_dim - 1
=======
        self.fixed_gauge_index = self.fixed_gauge_flat_start
        self.earlier_gauge_index = self.fixed_gauge_index + embedding_dim - 1
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
        preceding_gauge_coords = (
=======
        fixed_gauge_coords = (
            flat_weight[
                self.fixed_gauge_flat_start :
                self.earlier_gauge_flat_start - 1
            ]
            - flat_weight[self.earlier_gauge_flat_start - 1]
        )
        earlier_gauge_coords = (
            self.gauge_basis.transpose(0, 1)
            @ flat_weight[
                self.earlier_gauge_flat_start :
                self.earlier_gauge_flat_start + embedding_dim
            ]
        )
        preceding_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        compact_weight = torch.cat(
            (
                flat_weight[: self.earlier_gauge_flat_start],
                earlier_gauge_coords,
=======
        compact_weight = torch.cat(
            (
                flat_weight[: self.fixed_gauge_flat_start],
                fixed_gauge_coords,
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
        fixed_gauge_row = F.pad(
            self.weight[self.fixed_gauge_index : self.earlier_gauge_index],
            (0, 1),
        )
        earlier_gauge_row = (
            self.gauge_basis
            @ self.weight[self.earlier_gauge_index : self.preceding_gauge_index]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        flat_weight = torch.cat(
            (
                self.weight[: self.earlier_gauge_index],
                earlier_gauge_row,
=======
        flat_weight = torch.cat(
            (
                self.weight[: self.fixed_gauge_index],
                fixed_gauge_row,
                earlier_gauge_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            flat_weight = full_weight.flatten()
            earlier_gauge_coords = (
=======
            fixed_anchor = full_weight[-12, -1].clone()
            full_weight[-12].sub_(fixed_anchor)
            anchor = full_weight[-1, -1].clone()
            full_weight[-1].sub_(anchor)
            flat_weight = full_weight.flatten()
            fixed_gauge_coords = flat_weight[
                module.fixed_gauge_flat_start :
                module.earlier_gauge_flat_start - 1
            ]
            earlier_gauge_coords = (
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_weight = torch.cat(
                (
                    flat_weight[: module.earlier_gauge_flat_start],
                    earlier_gauge_coords,
=======
            compact_weight = torch.cat(
                (
                    flat_weight[: module.fixed_gauge_flat_start],
                    fixed_gauge_coords,
                    earlier_gauge_coords,
>>>>>>> REPLACE