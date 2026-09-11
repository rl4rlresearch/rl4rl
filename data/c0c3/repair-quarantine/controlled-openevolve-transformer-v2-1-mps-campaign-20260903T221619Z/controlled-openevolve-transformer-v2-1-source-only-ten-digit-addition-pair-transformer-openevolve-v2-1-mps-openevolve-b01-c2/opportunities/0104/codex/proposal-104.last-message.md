MECHANISM: Coordinate-1 attention-LayerNorm scale quotient

HYPOTHESIS: Extending the verified 1,534-parameter Helmert query-row design by fixing attention LayerNorm coordinate 1 will produce a 1,533-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the successful orthonormal gauge for second-head query row 6, then anchor LayerNorm scale coordinate 1 while retaining coordinates 0 and 2 as learned scales and exposing the reconstructed dense scale vector to the QKV quotient optimizer.

EVIDENCE: The Helmert query-row design achieved 99.89% at 1,534 parameters. Previous sixth-scale attempts tested coordinates 0 and 2 and reached 97.41% and 94.20%, leaving coordinate 1 as the only untested learned attention-LayerNorm scale and the most direct coordinate-specific test.

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
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
=======
        # Retain the verified coordinate charts and use the successful Helmert
        # chart for second-head query row 6.
        self.orthonormal_rows = (head_dim + 2,)
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
        reduced_rows = set(self.gauged_rows) | set(self.orthonormal_rows)
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in reduced_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = (
            self.out_features * d_model
            - len(self.gauged_rows)
            - len(self.orthonormal_rows)
        )
        self.weight = nn.Parameter(base.weight.new_empty(retained))

        basis = torch.zeros(d_model, d_model - 1)
        for column in range(d_model - 1):
            scale = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / scale
            basis[column + 1, column] = -(column + 1) / scale
        self.register_buffer("query_row_basis", basis, persistent=False)

        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
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
        gauged_stop = len(self.gauged_rows) * row_width
        orthonormal_stop = (
            gauged_stop + len(self.orthonormal_rows) * row_width
        )
        gauged = torch.cat(
            (
                self.weight[:gauged_stop].view(
                    len(self.gauged_rows), row_width
                ),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        orthonormal_coordinates = self.weight[
            gauged_stop:orthonormal_stop
        ].view(len(self.orthonormal_rows), row_width)
        orthonormal = (
            orthonormal_coordinates @ self.query_row_basis.transpose(0, 1)
        )
        ungauged = self.weight[orthonormal_stop:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        gauged_index = 0
        orthonormal_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
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
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
        return F.layer_norm(x, self.normalized_shape, weight, None, 1e-5)
=======
class AnchoredScaleLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        # Coordinates 0 and 2 remain learned; coordinate 1 joins the five
        # previously anchored scales.
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def dense_weight(self) -> torch.Tensor:
        return torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:],
                self.weight.new_ones(5),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, self.normalized_shape, self.dense_weight(), None, 1e-5
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
                nn.init.zeros_(module.bias)
=======
                # With unit initial LayerNorm scales, both charts discard only
                # the common input coefficient annihilated by normalization.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                orthonormal = (
                    full[list(module.orthonormal_rows)]
                    @ module.query_row_basis
                )
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat(
                        (
                            gauged[:, :-1].flatten(),
                            orthonormal.flatten(),
                            ungauged.flatten(),
                        )
                    )
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        scales = torch.cat(
            (
                normalization.weight,
                normalization.weight.new_ones(
                    module.in_features - normalization.weight.numel()
                ),
            )
        )
=======
        scales = normalization.dense_weight()
>>>>>>> REPLACE