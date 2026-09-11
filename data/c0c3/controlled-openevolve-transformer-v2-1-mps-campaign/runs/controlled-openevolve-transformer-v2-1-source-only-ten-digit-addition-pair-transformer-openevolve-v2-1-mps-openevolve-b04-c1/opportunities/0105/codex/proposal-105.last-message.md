MECHANISM: Optimization-aware Q/K gauge rollback with farthest-lag edge pruning

HYPOTHESIS: Restoring the unbiased head’s first-row scale coordinate while fixing both heads’ farthest-lag biases at zero will produce a 845-parameter model with at least 99% accuracy, because it restores the better-conditioned 847-parameter Q/K optimization geometry and removes only two biases governing the single maximum-distance attention edge.

INTENDED_EDIT: Undo the unbiased-head scale anchor that reduced training margin, then shorten each head’s learned relative-bias vector by one coordinate and reconstruct the unused maximum-lag entries as zeros.

EVIDENCE: The seven-unit GLU reached 99.94% at 847 parameters, whereas adding the first unbiased-head scale gauge reduced accuracy to 99.13% and adding the second fell to 98.41%; this motivates restoring the demonstrated optimizer geometry and seeking the net reduction in the sparsest-used relative-attention coordinates.

<<<<<<< SEARCH
                        d_model - 3
                        if row in (
                            self.biased_shear_gauge_row,
                            self.shear_gauge_row,
                        )
                        else (
                            d_model - 2
                            if row in (
                                self.biased_second_scale_gauge_row,
                                self.rotation_gauge_row,
                            )
=======
                        d_model - 3
                        if row == self.biased_shear_gauge_row
                        else (
                            d_model - 2
                            if row in (
                                self.biased_second_scale_gauge_row,
                                self.shear_gauge_row,
                                self.rotation_gauge_row,
                            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "unbiased_scale_anchor", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "unbiased_scale_value", torch.tensor(0.0)
        )

=======
>>>>>>> REPLACE

<<<<<<< SEARCH
            unbiased_scale_candidates = [
                index
                for index in range(1, self.d_model)
                if index != pivot
            ]
            unbiased_scale_anchor = max(
                unbiased_scale_candidates,
                key=lambda index: float(
                    transformed[q0, index].abs().item()
                ),
            )
            self.unbiased_scale_anchor.fill_(unbiased_scale_anchor)
            self.unbiased_scale_value.copy_(
                transformed[q0, unbiased_scale_anchor]
            )
            unbiased_coordinate_indices = [
                index
                for index in unbiased_scale_candidates
                if index != unbiased_scale_anchor
            ]

=======
            # Retain the remaining diagonal scale as a learned coordinate;
            # only the shear pivot is fixed at zero.

>>>>>>> REPLACE

<<<<<<< SEARCH
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(row[unbiased_coordinate_indices])
=======
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            anchor = int(self.unbiased_scale_anchor.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index not in (pivot, anchor)
            )
=======
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif row_index == self.shear_gauge_row:
            anchor = int(self.unbiased_scale_anchor.item())
            fixed[anchor] = self.unbiased_scale_value
        return fixed
=======
        return fixed
>>>>>>> REPLACE

<<<<<<< SEARCH
            elif row_index == self.shear_gauge_row:
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
                    self.unbiased_scale_anchor.reshape(1),
                    self.unbiased_scale_value.to(
                        dtype=coordinates.dtype
                    ).reshape(1),
                )
=======
            elif row_index == self.shear_gauge_row:
                coordinate_indices = torch.tensor(
                    self.ambient_coordinate_indices(row_index),
                    device=coordinates.device,
                    dtype=torch.long,
                )
                row = coordinates.new_zeros(self.d_model).scatter(
                    0, coordinate_indices, coordinates
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
        self.relative_bias_coordinates = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 0))
=======
        relative_bias = F.pad(self.relative_bias_coordinates, (1, 1))
>>>>>>> REPLACE