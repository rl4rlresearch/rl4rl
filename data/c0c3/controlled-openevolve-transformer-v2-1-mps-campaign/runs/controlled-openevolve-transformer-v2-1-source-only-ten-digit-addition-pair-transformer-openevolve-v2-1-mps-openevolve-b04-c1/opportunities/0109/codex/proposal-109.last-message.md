MECHANISM: Unbiased-head diagonal Q/K scale quotient

HYPOTHESIS: Combining the previously successful unbiased-head first-channel scale gauge with the current farthest-lag pruning will reduce the model from 845 to 844 parameters while retaining at least 99% accuracy, because each reduction independently exceeded the threshold and this gauge preserves the attention function class exactly.

INTENDED_EDIT: Fix the largest surviving coefficient of the unbiased head’s first query row at initialization, omit it from learned coordinates, reconstruct it during forward passes, and retain ambient-space gauge optimization.

EVIDENCE: The same unbiased-head scale anchor achieved 99.13% at 846 parameters, while the current two-edge relative-bias pruning achieved 99.84% at 845 parameters; composing them tests a known exact gauge without the reconstruction overhead associated with the three timed-out relative-bias experiments.

<<<<<<< SEARCH
                        d_model - 3
                        if row == self.biased_shear_gauge_row
                        else (
=======
                        d_model - 3
                        if row in (
                            self.biased_shear_gauge_row,
                            self.shear_gauge_row,
                        )
                        else (
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
            # Retain the remaining diagonal scale as a learned coordinate;
            # only the shear pivot is fixed at zero.

            for row_index, (coordinates, row) in enumerate(
=======
            # Quotient the remaining first-channel diagonal scale by fixing
            # its largest surviving query coefficient. Inverse scaling of the
            # corresponding key channel preserves every attention logit.
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
                coordinate_indices = torch.tensor(
                    self.ambient_coordinate_indices(row_index),
                    device=coordinates.device,
                    dtype=torch.long,
                )
                row = coordinates.new_zeros(self.d_model).scatter(
                    0, coordinate_indices, coordinates
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