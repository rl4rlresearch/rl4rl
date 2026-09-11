MECHANISM: Unbiased-head second-channel diagonal Q/K scale gauge

HYPOTHESIS: Fixing the largest surviving coefficient of the unbiased head’s second query row will reduce the model from 846 to 845 learned parameters while retaining at least 99% accuracy, because scaling that bias-free query channel while inversely scaling its key channel preserves every attention logit.

INTENDED_EDIT: Anchor the unbiased head’s second query row at its existing shear pivot, omit that coordinate from training, reconstruct it from a checkpointed buffer, and include the fixed value in ambient AdamW updates.

EVIDENCE: The analogous second-channel scale quotient in the biased head retained 99.93% accuracy, and fixing the unbiased head’s first-channel scale produced the current 846-parameter model at 99.90%; the remaining unbiased query channel admits the same exact symmetry.

<<<<<<< SEARCH
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
=======
                        if row in (
                            self.biased_shear_gauge_row,
                            self.shear_gauge_row,
                            self.rotation_gauge_row,
                        )
                        else (
                            d_model - 2
                            if row == self.biased_second_scale_gauge_row
                            else d_model - 1
                        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "shear_scale_value", torch.tensor(0.0)
        )
=======
        self.register_buffer(
            "shear_scale_value", torch.tensor(0.0)
        )
        self.register_buffer(
            "rotation_scale_value", torch.tensor(0.0)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.shear_pivot.fill_(pivot)
            shear = transformed[q0, pivot] / transformed[q1, pivot]
=======
            self.shear_pivot.fill_(pivot)
            self.rotation_scale_value.copy_(transformed[q1, pivot])
            shear = transformed[q0, pivot] / transformed[q1, pivot]
>>>>>>> REPLACE

<<<<<<< SEARCH
                elif row_index == self.rotation_gauge_row:
                    coordinates.copy_(row[2:])
=======
                elif row_index == self.rotation_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[2:pivot], row[pivot + 1 :]))
                    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
=======
        if row_index == self.rotation_gauge_row:
            pivot = int(self.shear_pivot.item())
            return tuple(
                index
                for index in range(2, self.d_model)
                if index != pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif row_index == self.shear_gauge_row:
            anchor = int(self.shear_scale_anchor.item())
            fixed[anchor] = self.shear_scale_value
        return fixed
=======
        elif row_index == self.shear_gauge_row:
            anchor = int(self.shear_scale_anchor.item())
            fixed[anchor] = self.shear_scale_value
        elif row_index == self.rotation_gauge_row:
            pivot = int(self.shear_pivot.item())
            fixed[pivot] = self.rotation_scale_value
        return fixed
>>>>>>> REPLACE

<<<<<<< SEARCH
            elif row_index == self.rotation_gauge_row:
                row = F.pad(coordinates, (2, 0))
=======
            elif row_index == self.rotation_gauge_row:
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
                    self.shear_pivot.reshape(1),
                    self.rotation_scale_value.to(
                        dtype=coordinates.dtype
                    ).reshape(1),
                )
>>>>>>> REPLACE