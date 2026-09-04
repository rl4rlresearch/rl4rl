MECHANISM: Orthonormal second-head query-row quotient

HYPOTHESIS: Gauging query row 6 in a Helmert zero-mean basis will produce a 1,534-parameter model with at least 99% accuracy by avoiding the poorly conditioned coordinate pivot that reached 98.89%.

INTENDED_EDIT: Add query row 6 to the normalized-input gauge, represent that row in an orthonormal seven-dimensional complement, and retain the verified coordinate gauges and dense updates for rows 15, 20, and 23.

EVIDENCE: Query row 6 nearly met the threshold with reduced-coordinate AdamW at 98.89%, while dense-coordinate recovery reached 98.52%; the current 1,535-parameter model reaches 99.82% while successfully using Helmert quotient coordinates for residual projections, motivating a better-conditioned chart rather than another positional anchor.

<<<<<<< SEARCH
        # Retain the verified query-row-7 design and every key and value
        # gauge. Final rows 15 and 23 and value row 20 use recovered
        # dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 3,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
=======
        # Retain every verified coordinate gauge and add second-head query row
        # 6 in an orthonormal chart. Sensitive rows 15, 20, and 23 continue to
        # use recovered dense-coordinate AdamW moments.
        self.gauged_rows = (
            0,
            1,
            2,
            3,
            head_dim,
            head_dim + 1,
            head_dim + 2,
            head_dim + 3,
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + head_dim,
            d_model + head_dim + 1,
            d_model + head_dim + 2,
            d_model + head_dim + 3,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + 2,
            2 * d_model + 3,
            2 * d_model + head_dim,
            2 * d_model + head_dim + 1,
            2 * d_model + head_dim + 2,
            2 * d_model + head_dim + 3,
        )
        self.orthonormal_rows = (head_dim + 2,)
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # A Helmert basis supplies a balanced complement to the normalized
        # input's common-coefficient null direction for the new query row.
        basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("input_basis", basis, persistent=False)

        # Consume the same constructor RNG stream as the replaced nn.Linear.
>>>>>>> REPLACE

<<<<<<< SEARCH
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(len(self.gauged_rows), row_width),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )
=======
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        reduced_gauged = self.weight[:split].view(
            len(self.gauged_rows), row_width
        )
        gauged_parts = []
        for row, reduced_row in zip(self.gauged_rows, reduced_gauged):
            if row in self.orthonormal_rows:
                gauged_parts.append(self.input_basis @ reduced_row)
            else:
                gauged_parts.append(
                    torch.cat((reduced_row, reduced_row.new_zeros(1)))
                )
        gauged = torch.stack(gauged_parts)
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves both selected row functions.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
=======
                # Attention scales initialize to one. Coordinate-chart rows
                # subtract their omitted coefficient, while the new query row
                # is projected into the Helmert zero-mean complement.
                gauged = []
                for row in module.gauged_rows:
                    row_weight = full[row]
                    if row in module.orthonormal_rows:
                        gauged.append(
                            module.input_basis.transpose(0, 1) @ row_weight
                        )
                    else:
                        gauged.append(row_weight[:-1] - row_weight[-1])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat(
                        (torch.stack(gauged).flatten(), ungauged.flatten())
                    )
                )
>>>>>>> REPLACE