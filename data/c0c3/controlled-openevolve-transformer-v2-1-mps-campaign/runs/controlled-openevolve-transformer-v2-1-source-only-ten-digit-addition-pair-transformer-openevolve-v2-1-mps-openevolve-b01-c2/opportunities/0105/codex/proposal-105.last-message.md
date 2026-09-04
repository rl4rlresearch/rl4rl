MECHANISM: Helmert-chart final-query-row LayerNorm quotient

HYPOTHESIS: Adding second-head query row 6 through an orthonormal Helmert chart will reduce the verified 1,513-parameter model to 1,512 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Keep the 23 verified QKV rows in their existing last-coordinate charts, represent query row 6 in a seven-dimensional Helmert basis, and preserve the verified dense-coordinate updates for rows 15, 20, and 23.

EVIDENCE: The position-scalar model achieved 99.95% at 1,513 parameters; query row 6 previously came within 0.11 percentage points of the threshold, and the proposed Helmert experiment was never tested because its patch matched multiple locations.

<<<<<<< SEARCH
        # Gauge the final second-head query row while retaining every verified
        # key and value gauge. Sensitive rows 15, 20, and 23 use recovered
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
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        # Fix local query coordinate 2 in both heads. Key and value biases
        # remain absent through their existing exact attention gauges.
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
=======
        # Retain the verified last-coordinate charts, and represent the
        # remaining second-head query row in an orthonormal zero-mean chart.
        # Sensitive rows 15, 20, and 23 keep dense-coordinate AdamW moments.
        self.anchored_rows = (
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
        self.orthonormal_rows = (head_dim + 2,)
        self.gauged_rows = self.anchored_rows + self.orthonormal_rows
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        # Fix local query coordinate 2 in both heads. Key and value biases
        # remain absent through their existing exact attention gauges.
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )

        basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("input_basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        rows = []
        gauged_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)

        query_bias_parts = []
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        anchored_split = len(self.anchored_rows) * row_width
        gauged_split = len(self.gauged_rows) * row_width
        anchored = torch.cat(
            (
                self.weight[:anchored_split].view(
                    len(self.anchored_rows), row_width
                ),
                self.weight.new_zeros(len(self.anchored_rows), 1),
            ),
            dim=1,
        )
        orthonormal_reduced = self.weight[
            anchored_split:gauged_split
        ].view(len(self.orthonormal_rows), row_width)
        orthonormal = (
            orthonormal_reduced @ self.input_basis.transpose(0, 1)
        )
        ungauged = self.weight[gauged_split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        anchored_index = 0
        orthonormal_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.anchored_rows:
                rows.append(anchored[anchored_index])
                anchored_index += 1
            elif row in self.orthonormal_rows:
                rows.append(orthonormal[orthonormal_index])
                orthonormal_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)

        query_bias_parts = []
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # LayerNorm scales initialize to one, so subtracting each
                # omitted coefficient preserves every selected row function.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # LayerNorm scales initialize to one. The anchored rows use
                # their verified final-zero representatives, while query row
                # 6 uses orthonormal coordinates for the same exact quotient.
                anchored = full[list(module.anchored_rows)].clone()
                anchored[:, :-1].sub_(anchored[:, -1:])
                orthonormal = (
                    full[list(module.orthonormal_rows)] @ module.input_basis
                )
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat(
                        (
                            anchored[:, :-1].flatten(),
                            orthonormal.flatten(),
                            ungauged.flatten(),
                        )
                    )
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE