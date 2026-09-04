MECHANISM: Single-channel GLU value/output scale quotient

HYPOTHESIS: Gauge-fixing one GLU value-row coefficient will reduce the model from 845 to 844 learned parameters while retaining at least 99% accuracy, because inverse scaling of the corresponding learned output-projection column preserves the MLP function exactly.

INTENDED_EDIT: Anchor the largest initialized coefficient of the first GLU value row, reconstruct it from a checkpointed buffer, and optimize its remaining coordinates in ambient space.

EVIDENCE: The seven-unit GLU reached 99.94% accuracy at 847 parameters, while simultaneously anchoring all seven value rows collapsed to 45.12%; isolating one exact scale quotient tests whether that failure arose from compounding seven poorly conditioned gauge constraints.

<<<<<<< SEARCH
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(in_features - 1))
                for _ in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))
=======
        self.value_scale_gauge_row = 0
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        in_features - 2
                        if row == self.value_scale_gauge_row
                        else in_features - 1
                    )
                )
                for row in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "value_scale_anchor", torch.tensor(1, dtype=torch.long)
        )
        self.register_buffer(
            "value_scale_value", torch.tensor(0.0)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])
=======
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            anchor = int(
                torch.argmax(
                    transformed[self.value_scale_gauge_row, 1:].abs()
                ).item()
            ) + 1
            self.value_scale_anchor.fill_(anchor)
            self.value_scale_value.copy_(
                transformed[self.value_scale_gauge_row, anchor]
            )
            for row_index, (coordinates, row) in enumerate(
                zip(self.coordinates, transformed)
            ):
                if row_index == self.value_scale_gauge_row:
                    coordinates.copy_(
                        torch.cat((row[1:anchor], row[anchor + 1 :]))
                    )
                else:
                    coordinates.copy_(row[1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def reset_parameters(self) -> None:
        conceptual_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        weight = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, weight, self.bias)
=======
    def reset_parameters(self) -> None:
        conceptual_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
        nn.init.zeros_(self.bias)

    def ambient_coordinate_indices(
        self, row_index: int
    ) -> Tuple[int, ...]:
        if row_index == self.value_scale_gauge_row:
            anchor = int(self.value_scale_anchor.item())
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
        if row_index == self.value_scale_gauge_row:
            anchor = int(self.value_scale_anchor.item())
            fixed[anchor] = self.value_scale_value
        return fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = []
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index == self.value_scale_gauge_row:
                coordinate_indices = torch.tensor(
                    self.ambient_coordinate_indices(row_index),
                    device=coordinates.device,
                    dtype=torch.long,
                )
                row = coordinates.new_zeros(self.in_features).scatter(
                    0, coordinate_indices, coordinates
                )
                row = row.scatter(
                    0,
                    self.value_scale_anchor.reshape(1),
                    self.value_scale_value.to(
                        dtype=coordinates.dtype
                    ).reshape(1),
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