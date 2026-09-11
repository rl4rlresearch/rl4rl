MECHANISM: Biased-head second-channel diagonal Q/K scale gauge

HYPOTHESIS: Fixing one nonzero coefficient of the biased head’s second query row will reduce the model from 867 to 866 learned parameters while retaining at least 99% accuracy, because scaling that query channel while inversely scaling its key channel preserves every attention logit and leaves the one-axis query bias unchanged.

INTENDED_EDIT: Preserve the initialized largest coefficient of the biased head’s second query row as a checkpointed buffer, omit it from learned coordinates, reconstruct it during forward passes, and include the fixed value in ambient AdamW updates.

EVIDENCE: Fixing the corresponding scale of the biased head’s first query channel produced the current 867-parameter model at 99.96% accuracy; the remaining independently scalable, bias-free second channel admits the same exact quotient.

<<<<<<< SEARCH
        # The first head's one-axis query bias is preserved by an upper
        # shear. The unbiased second head additionally permits the existing
        # rotation and independent shear gauges.
        self.biased_shear_gauge_row = 0
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
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
=======
        # The first head's one-axis query bias is preserved by an upper
        # shear and by scaling its bias-free second channel. The unbiased
        # second head additionally permits rotation and shear gauges.
        self.biased_shear_gauge_row = 0
        self.biased_second_scale_gauge_row = 1
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
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
                    )
                )
                for row in range(self.out_features)
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "biased_scale_value", torch.tensor(0.0)
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
=======
        self.register_buffer(
            "biased_scale_value", torch.tensor(0.0)
        )
        self.register_buffer(
            "biased_second_scale_value", torch.tensor(0.0)
        )
        self.register_buffer(
            "shear_pivot", torch.tensor(2, dtype=torch.long)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.biased_shear_pivot.fill_(biased_pivot)
            biased_shear = (
                transformed[biased_q0, biased_pivot]
                / transformed[biased_q1, biased_pivot]
            )
=======
            self.biased_shear_pivot.fill_(biased_pivot)
            self.biased_second_scale_value.copy_(
                transformed[biased_q1, biased_pivot]
            )
            biased_shear = (
                transformed[biased_q0, biased_pivot]
                / transformed[biased_q1, biased_pivot]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                if row_index == self.biased_shear_gauge_row:
                    coordinates.copy_(row[biased_coordinate_indices])
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
=======
                if row_index == self.biased_shear_gauge_row:
                    coordinates.copy_(row[biased_coordinate_indices])
                elif row_index == self.biased_second_scale_gauge_row:
                    coordinates.copy_(
                        torch.cat(
                            (
                                row[1:biased_pivot],
                                row[biased_pivot + 1 :],
                            )
                        )
                    )
                elif row_index == self.shear_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
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
        if row_index == self.biased_second_scale_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
        if row_index == self.shear_gauge_row:
            pivot = int(self.shear_pivot.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != pivot
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.biased_shear_gauge_row:
            anchor = int(self.biased_scale_anchor.item())
            fixed[anchor] = self.biased_scale_value
        return fixed
=======
        if row_index == self.biased_shear_gauge_row:
            anchor = int(self.biased_scale_anchor.item())
            fixed[anchor] = self.biased_scale_value
        elif row_index == self.biased_second_scale_gauge_row:
            pivot = int(self.biased_shear_pivot.item())
            fixed[pivot] = self.biased_second_scale_value
        return fixed
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformed_rows = []
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
=======
        transformed_rows = []
        biased_pivot = int(self.biased_shear_pivot.item())
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
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
            elif row_index == self.biased_second_scale_gauge_row:
                split = biased_pivot - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        self.biased_second_scale_value.to(
                            dtype=coordinates.dtype
                        ).reshape(1),
                        coordinates[split:],
                    )
                )
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
>>>>>>> REPLACE