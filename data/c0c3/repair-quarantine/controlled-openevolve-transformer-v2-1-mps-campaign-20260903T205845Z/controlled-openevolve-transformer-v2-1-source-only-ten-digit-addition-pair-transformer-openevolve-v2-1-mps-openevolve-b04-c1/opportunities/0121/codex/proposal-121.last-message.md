MECHANISM: Per-unit GLU scaling-gauge fixing

HYPOTHESIS: Fixing one nonzero value-projection coordinate per GLU unit removes seven exact value/output scaling redundancies, reducing parameters from 847 to 840 while retaining at least 99% accuracy.

INTENDED_EDIT: Store one initialization-preserving value-row coordinate as a buffer for each GLU unit, reconstruct it during forward, and optimize only the remaining ambient coordinates.

EVIDENCE: The independent seven-unit GLU achieved 99.94% accuracy at 847 parameters, while sharing value and gate features collapsed accuracy to 5.08%; this patch preserves every independent value and gate projection and removes only exact GLU scaling gauges.

<<<<<<< SEARCH
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(in_features - 1))
                for _ in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
=======
        self.value_rows = out_features // 2
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        in_features - 2
                        if row < self.value_rows
                        else in_features - 1
                    )
                )
                for row in range(out_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features))

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))
        self.register_buffer(
            "scale_pivots",
            torch.ones(self.value_rows, dtype=torch.long),
        )
        self.register_buffer(
            "scale_values", torch.zeros(self.value_rows)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])

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
    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            for row_index, (coordinates, row) in enumerate(
                zip(self.coordinates, transformed)
            ):
                if row_index < self.value_rows:
                    pivot = int(
                        torch.argmax(row[1:].abs()).item()
                    ) + 1
                    self.scale_pivots[row_index] = pivot
                    self.scale_values[row_index].copy_(row[pivot])
                    coordinates.copy_(
                        torch.cat((row[1:pivot], row[pivot + 1 :]))
                    )
                else:
                    coordinates.copy_(row[1:])

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
        if row_index < self.value_rows:
            pivot = int(self.scale_pivots[row_index].item())
            return tuple(
                index
                for index in range(1, self.in_features)
                if index != pivot
            )
        return tuple(range(1, self.in_features))

    def ambient_fixed_coordinates(
        self, row_index: int
    ) -> torch.Tensor:
        fixed = self.coordinates[row_index].new_zeros(
            self.in_features
        )
        if row_index < self.value_rows:
            pivot = int(self.scale_pivots[row_index].item())
            fixed[pivot] = self.scale_values[row_index]
        return fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed_rows = []
        for row_index, coordinates in enumerate(self.coordinates):
            if row_index < self.value_rows:
                pivot = int(self.scale_pivots[row_index].item())
                split = pivot - 1
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