MECHANISM: Biased-head diagonal Q/K scale gauge

HYPOTHESIS: Fixing one nonzero coefficient of the biased head’s first query row will reduce the model from 868 to 867 learned parameters while retaining at least 99% accuracy, because scaling that query channel and its bias while inversely scaling the corresponding key channel preserves every attention logit.

INTENDED_EDIT: Preserve the initialized value of the largest surviving query coefficient as a checkpointed buffer, omit it from learned coordinates, reconstruct it during forward passes, and include the fixed base value in ambient AdamW updates.

EVIDENCE: The biased-head stabilizer shear retained 99.78% accuracy at 868 parameters; diagonal scaling is another exact Q/K change-of-basis symmetry that preserves the established attention capacity.

<<<<<<< SEARCH
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        d_model - 2
                        if row in (
                            self.biased_shear_gauge_row,
                            self.shear_gauge_row,
                            self.rotation_gauge_row,
                        )
                        else d_model - 1
                    )
                )
                for row in range(self.out_features)
            ]
        )
        self.register_buffer(
            "biased_shear_pivot", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
=======
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        d_model - 3
                        if row == self.biased_shear_gauge_row
                        else (
                            d_model - 2
                            if row in (
                                self.shear_gauge_row,
                                self.rotation_gauge_row,
                            )
                            else d_model - 1
                        )
                    )
                )
                for row in range(self.out_features)
            ]
        )
        self.register_buffer(
            "biased_shear_pivot", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "biased_scale_anchor", torch.tensor(2, dtype=torch.long)
        )
        self.register_buffer(
            "biased_scale_value", torch.tensor(0.0)
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            transformed[biased_k1, 1:] = (
                transformed[biased_k1, 1:]
                + biased_shear * transformed[biased_k0, 1:]
            )

            q0 = self.qk_dim
=======
            transformed[biased_k1, 1:] = (
                transformed[biased_k1, 1:]
                + biased_shear * transformed[biased_k0, 1:]
            )

            biased_scale_candidates = [
                index
                for index in range(1, self.d_model)
                if index != biased_pivot
            ]
            biased_scale_anchor = max(
                biased_scale_candidates,
                key=lambda index: float(
                    transformed[biased_q0, index].abs().item()
                ),
            )
            self.biased_scale_anchor.fill_(biased_scale_anchor)
            self.biased_scale_value.copy_(
                transformed[biased_q0, biased_scale_anchor]
            )
            biased_coordinate_indices = [
                index
                for index in biased_scale_candidates
                if index != biased_scale_anchor
            ]

            q0 = self.qk_dim
>>>>>>> REPLACE

<<<<<<< SEARCH
                if row_index == self.biased_shear_gauge_row:
                    coordinates.copy_(
                        torch.cat(
                            (
                                row[1:biased_pivot],
                                row[biased_pivot + 1 :],
                            )
                        )
                    )
=======
                if row_index == self.biased_shear_gauge_row:
                    coordinates.copy_(row[biased_coordinate_indices])
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.biased_shear_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
        if row_index == self.shear_gauge_row:
=======
        if row_index == self.biased_shear_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            anchor = int(self.biased_scale_anchor.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index not in (pivot, anchor)
            )
        if row_index == self.shear_gauge_row:
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
        return tuple(range(1, self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
        return tuple(range(1, self.d_model))

    def ambient_fixed_coordinates(
        self, row_index: int
    ) -> torch.Tensor:
        fixed = self.coordinates[row_index].new_zeros(self.d_model)
        if row_index == self.biased_shear_gauge_row:
            anchor = int(self.biased_scale_anchor.item())
            fixed[anchor] = self.biased_scale_value
        return fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformed_rows = []
        biased_pivot = int(self.biased_shear_pivot.item())
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index == self.biased_shear_gauge_row:
                split = biased_pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        coordinates.new_zeros(1),
                        coordinates[split:],
                    )
                )
=======
        transformed_rows = []
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index == self.biased_shear_gauge_row:
                coordinate_indices = torch.tensor(
                    self.ambient_coordinate_indices(row_index),
                    device=coordinates.device,
                    dtype=torch.long,
                )
                row = coordinates.new_zeros(self.d_model).scatter(
                    0, coordinate_indices, coordinates
                )
                row = row.scatter(
                    0,
                    self.biased_scale_anchor.reshape(1),
                    self.biased_scale_value.to(
                        dtype=coordinates.dtype
                    ).reshape(1),
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
            else:
                coordinate_indices = list(entry[3])
                padded = parameter.new_zeros(reflector.numel())
                padded[coordinate_indices] = parameter
                grad_padded = parameter.grad.new_zeros(reflector.numel())
                grad_padded[coordinate_indices] = parameter.grad
=======
            else:
                coordinate_indices = list(entry[3])
                padded = (
                    entry[4].clone()
                    if len(entry) > 4
                    else parameter.new_zeros(reflector.numel())
                )
                padded[coordinate_indices] = parameter
                grad_padded = parameter.grad.new_zeros(reflector.numel())
                grad_padded[coordinate_indices] = parameter.grad
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.qkv.reflector_norm_sq,
            block.attn.qkv.ambient_coordinate_indices(row_index),
        )
=======
            block.attn.qkv.reflector_norm_sq,
            block.attn.qkv.ambient_coordinate_indices(row_index),
            block.attn.qkv.ambient_fixed_coordinates(row_index),
        )
>>>>>>> REPLACE