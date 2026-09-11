MECHANISM: GLU value-channel scale quotient

HYPOTHESIS: Fixing one nonzero weight coordinate in each of the seven GLU value rows will reduce the model from 847 to 840 learned parameters while retaining at least 99% accuracy, because independently scaling a value row and inversely scaling its output-projection column leaves the learned MLP function unchanged.

INTENDED_EDIT: Gauge-fix the seven GLU value-row scales at their largest initialized coordinates, reconstruct those coordinates from checkpointed buffers, and optimize the remaining weights in ambient space.

EVIDENCE: The seven-unit GLU achieved 99.94% accuracy at 847 parameters while preserving seven independent output directions; the proposed change retains every direction and removes only seven independent continuous scaling redundancies between value rows and their corresponding output columns.

<<<<<<< SEARCH
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(in_features - 1))
                for _ in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        inv_sqrt = in_features ** -0.5
=======
        # The first half of the rows are GLU value channels. Each can be
        # rescaled while its corresponding output column is inversely scaled,
        # so fix one stable nonzero coordinate in every such row.
        self.scale_gauge_rows = out_features // 2
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        in_features - 2
                        if row < self.scale_gauge_rows
                        else in_features - 1
                    )
                )
                for row in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "scale_anchors",
            torch.zeros(self.scale_gauge_rows, dtype=torch.long),
        )
        self.register_buffer(
            "scale_values", torch.zeros(self.scale_gauge_rows)
        )

        inv_sqrt = in_features ** -0.5
>>>>>>> REPLACE

<<<<<<< SEARCH
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])
=======
            for row_index, (coordinates, row) in enumerate(
                zip(self.coordinates, transformed)
            ):
                if row_index < self.scale_gauge_rows:
                    anchor = int(
                        torch.argmax(row[1:].abs()).item()
                    ) + 1
                    self.scale_anchors[row_index] = anchor
                    self.scale_values[row_index] = row[anchor]
                    coordinates.copy_(
                        torch.cat((row[1:anchor], row[anchor + 1 :]))
                    )
                else:
                    coordinates.copy_(row[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        weight = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, weight, self.bias)
=======
    def ambient_coordinate_indices(
        self, row_index: int
    ) -> Tuple[int, ...]:
        if row_index < self.scale_gauge_rows:
            anchor = int(self.scale_anchors[row_index].item())
            return tuple(
                index
                for index in range(1, self.in_features)
                if index != anchor
            )
        return tuple(range(1, self.in_features))

    def ambient_fixed_coordinates(
        self, row_index: int
    ) -> torch.Tensor:
        fixed = self.coordinates[row_index].new_zeros(self.in_features)
        if row_index < self.scale_gauge_rows:
            anchor = int(self.scale_anchors[row_index].item())
            fixed[anchor] = self.scale_values[row_index]
        return fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = []
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index < self.scale_gauge_rows:
                anchor = int(self.scale_anchors[row_index].item())
                split = anchor - 1
                row = torch.cat(
                    (
                        coordinates.new_zeros(1),
                        coordinates[:split],
                        self.scale_values[row_index]
                        .to(dtype=coordinates.dtype)
                        .reshape(1),
                        coordinates[split:],
                    )
                )
            else:
                row = F.pad(coordinates, (1, 0))
            transformed_rows.append(row)

        weight = self._householder(
            torch.stack(transformed_rows, dim=0)
        )
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        (
            coordinates,
            block.mlp.fc1.reflector,
            block.mlp.fc1.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in block.mlp.fc1.coordinates
    ] + [
=======
    ] + [
        (
            coordinates,
            block.mlp.fc1.reflector,
            block.mlp.fc1.reflector_norm_sq,
            block.mlp.fc1.ambient_coordinate_indices(row_index),
            block.mlp.fc1.ambient_fixed_coordinates(row_index),
        )
        for block in model.blocks
        for row_index, coordinates in enumerate(
            block.mlp.fc1.coordinates
        )
    ] + [
>>>>>>> REPLACE