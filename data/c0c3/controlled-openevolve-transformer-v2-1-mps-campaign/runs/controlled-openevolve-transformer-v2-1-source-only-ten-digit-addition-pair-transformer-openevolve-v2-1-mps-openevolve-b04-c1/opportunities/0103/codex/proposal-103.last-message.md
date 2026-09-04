MECHANISM: Unbiased-head diagonal Q/K scale gauge

HYPOTHESIS: Fixing one nonzero coefficient of the unbiased head’s first query row will reduce the model from 847 to 846 learned parameters while retaining at least 99% accuracy, because scaling that query channel while inversely scaling its key channel preserves every attention logit.

INTENDED_EDIT: Anchor the largest surviving coefficient of the unbiased head’s first query row, omit it from trainable coordinates, reconstruct it from a checkpointed buffer, and reuse the existing ambient gauge optimizer.

EVIDENCE: The analogous two biased-head query-scale gauges retained 99.96% and 99.93% accuracy; unlike the value/output-scale attempt that timed out, this adds no optimizer entries and extends the already-successful Q/K gauge parameterization.

<<<<<<< SEARCH
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
=======
        # The first head's one-axis query bias is preserved by an upper
        # shear and by scaling its bias-free second channel. The unbiased
        # second head additionally permits rotation, shear, and scale gauges.
        self.biased_shear_gauge_row = 0
        self.biased_second_scale_gauge_row = 1
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
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
                    )
                )
                for row in range(self.out_features)
            ]
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
                zip(self.coordinates, transformed)
            ):
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
                zip(self.coordinates, transformed)
            ):
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