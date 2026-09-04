MECHANISM: Unbiased-head Q/K scale-gauge fixing

HYPOTHESIS: Fixing one nonzero query-weight coordinate in the unbiased attention head will reduce the verified model from 847 to 846 parameters while retaining at least 99% accuracy, because inverse scaling of the corresponding key channel preserves every attention logit exactly.

INTENDED_EDIT: Remove one learned coordinate from the second head’s first query row, preserve its initialized value as a buffer, and reconstruct the row while retaining all other query and key degrees of freedom.

EVIDENCE: The 847-parameter model reached 99.94% accuracy while already using two successful fixed-coordinate Q/K scale gauges in the biased head; unlike the failed GLU constraints, this extends that verified attention-gauge mechanism to the unbiased head.

<<<<<<< SEARCH
                        d_model - 3
                        if row == self.biased_shear_gauge_row
                        else (
                            d_model - 2
                            if row in (
                                self.biased_second_scale_gauge_row,
                                self.shear_gauge_row,
                                self.rotation_gauge_row,
                            )
                            else d_model - 1
                        )
=======
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
                            else d_model - 1
                        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )

        inv_sqrt = d_model ** -0.5
=======
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
        self.register_buffer(
            "unbiased_scale_anchor", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "unbiased_scale_value", torch.tensor(0.0)
        )

        inv_sqrt = d_model ** -0.5
>>>>>>> REPLACE

<<<<<<< SEARCH
            transformed[k1, 1:] = (
                transformed[k1, 1:] + shear * transformed[k0, 1:]
            )

            for row_index, (coordinates, row) in enumerate(
=======
            transformed[k1, 1:] = (
                transformed[k1, 1:] + shear * transformed[k0, 1:]
            )

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

            for row_index, (coordinates, row) in enumerate(
>>>>>>> REPLACE

<<<<<<< SEARCH
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
=======
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(row[unbiased_coordinate_indices])
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
=======
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            anchor = int(self.unbiased_scale_anchor.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index not in (pivot, anchor)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif row_index == self.biased_second_scale_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            fixed[pivot] = self.biased_second_scale_value
        return fixed
=======
        elif row_index == self.biased_second_scale_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            fixed[pivot] = self.biased_second_scale_value
        elif row_index == self.shear_gauge_row:
            anchor = int(self.unbiased_scale_anchor.item())
            fixed[anchor] = self.unbiased_scale_value
        return fixed
>>>>>>> REPLACE

<<<<<<< SEARCH
            elif row_index == self.shear_gauge_row:
                split = pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        coordinates.new_zeros(1),
                        coordinates[split:],
                    )
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
                row = row.scatter(
                    0,
                    self.unbiased_scale_anchor.reshape(1),
                    self.unbiased_scale_value.to(
                        dtype=coordinates.dtype
                    ).reshape(1),
                )
>>>>>>> REPLACE