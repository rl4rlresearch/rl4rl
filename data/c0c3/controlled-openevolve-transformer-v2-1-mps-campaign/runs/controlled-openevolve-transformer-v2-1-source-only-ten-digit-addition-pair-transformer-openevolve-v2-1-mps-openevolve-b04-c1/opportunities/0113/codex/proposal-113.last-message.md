MECHANISM: Key-side unbiased-head Q/K scale gauge

HYPOTHESIS: Fixing one nonzero coordinate of the unbiased head’s first key row will produce a 844-parameter model with at least 99% accuracy, because it removes the same exact diagonal Q/K scaling redundancy whose query-side gauge previously achieved 99.13%, while testing a different optimization chart from the query-side composition that timed out.

INTENDED_EDIT: Anchor the largest initialized coefficient of the unbiased head’s first key row, omit it from learned coordinates, reconstruct it with a vectorized concatenation, and optimize the remaining coordinates in ambient space.

EVIDENCE: The unbiased-head first-query scale gauge met the threshold at 846 parameters, establishing that this exact symmetry can be quotient-fixed; the current farthest-lag-pruned model has substantially higher 99.84% accuracy, motivating the dual key-side gauge as an informative 844-parameter test with different optimization geometry.

<<<<<<< SEARCH
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
        self.shear_gauge_row = qk_dim
        self.rotation_gauge_row = qk_dim + 1
        self.unbiased_key_scale_gauge_row = n_head * qk_dim + qk_dim
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
                                self.unbiased_key_scale_gauge_row,
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
            "unbiased_key_scale_anchor", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "unbiased_key_scale_value", torch.tensor(0.0)
        )

        inv_sqrt = d_model ** -0.5
>>>>>>> REPLACE

<<<<<<< SEARCH
            transformed[k1, 1:] = (
                transformed[k1, 1:] + shear * transformed[k0, 1:]
            )

            # Retain the remaining diagonal scale as a learned coordinate;
            # only the shear pivot is fixed at zero.

            for row_index, (coordinates, row) in enumerate(
=======
            transformed[k1, 1:] = (
                transformed[k1, 1:] + shear * transformed[k0, 1:]
            )

            key_scale_anchor = int(
                torch.argmax(transformed[k0, 1:].abs()).item()
            ) + 1
            self.unbiased_key_scale_anchor.fill_(key_scale_anchor)
            self.unbiased_key_scale_value.copy_(
                transformed[k0, key_scale_anchor]
            )

            for row_index, (coordinates, row) in enumerate(
>>>>>>> REPLACE

<<<<<<< SEARCH
                elif row_index == self.rotation_gauge_row:
                    coordinates.copy_(row[2:])
                else:
                    coordinates.copy_(row[1:])
=======
                elif row_index == self.rotation_gauge_row:
                    coordinates.copy_(row[2:])
                elif row_index == self.unbiased_key_scale_gauge_row:
                    coordinates.copy_(
                        torch.cat(
                            (
                                row[1:key_scale_anchor],
                                row[key_scale_anchor + 1 :],
                            )
                        )
                    )
                else:
                    coordinates.copy_(row[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
        return tuple(range(1, self.d_model))
=======
        if row_index == self.rotation_gauge_row:
            return tuple(range(2, self.d_model))
        if row_index == self.unbiased_key_scale_gauge_row:
            anchor = int(self.unbiased_key_scale_anchor.item())
            return tuple(
                index
                for index in range(1, self.d_model)
                if index != anchor
            )
        return tuple(range(1, self.d_model))
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
        elif row_index == self.unbiased_key_scale_gauge_row:
            anchor = int(self.unbiased_key_scale_anchor.item())
            fixed[anchor] = self.unbiased_key_scale_value
        return fixed
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformed_rows = []
        biased_pivot = int(self.biased_shear_pivot.item())
        pivot = int(self.shear_pivot.item())
        for row_index, coordinates in enumerate(self.coordinates):
=======
        transformed_rows = []
        biased_pivot = int(self.biased_shear_pivot.item())
        pivot = int(self.shear_pivot.item())
        key_scale_anchor = int(self.unbiased_key_scale_anchor.item())
        for row_index, coordinates in enumerate(self.coordinates):
>>>>>>> REPLACE

<<<<<<< SEARCH
            elif row_index == self.rotation_gauge_row:
                row = F.pad(coordinates, (2, 0))
            else:
                row = F.pad(coordinates, (1, 0))
=======
            elif row_index == self.rotation_gauge_row:
                row = F.pad(coordinates, (2, 0))
            elif row_index == self.unbiased_key_scale_gauge_row:
                split = key_scale_anchor - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        self.unbiased_key_scale_value.to(
                            dtype=coordinates.dtype
                        ).reshape(1),
                        coordinates[split:],
                    )
                )
            else:
                row = F.pad(coordinates, (1, 0))
>>>>>>> REPLACE