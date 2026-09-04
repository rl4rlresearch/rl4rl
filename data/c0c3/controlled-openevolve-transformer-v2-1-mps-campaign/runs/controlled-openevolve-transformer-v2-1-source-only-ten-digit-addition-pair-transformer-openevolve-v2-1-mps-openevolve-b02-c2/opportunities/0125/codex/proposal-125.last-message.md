MECHANISM: Attention-projection residual-uniform gauge quotient

HYPOTHESIS: Reproducing the verified 1,580-parameter design and centering the first attention output-projection column will yield 1,579 parameters with at least 99% accuracy, because the removed component adds only a position-dependent uniform residual shift canceled by the following and final LayerNorms.

INTENDED_EDIT: Apply the qualified seven-key and nonadjacent four-column `fc2` compactions, then represent the first attention projection column in a seven-dimensional zero-mean basis.

EVIDENCE: The seven-key, nonadjacent four-`fc2`-column design achieved 99.30% accuracy at 1,580 parameters; successful `fc2` column reductions establish the same residual-uniform gauge, while the failed eighth-key and adjacent-fourth-`fc2` experiments motivate testing it in the independent attention projection.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with five key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with seven key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[key_start + 2 : second_head_start - 1],
                linear.weight[second_head_start + 2 : -2],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[
                    second_head_start + 2 : second_head_start + 3
                ],
                linear.weight[second_head_start + 4 : -2],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * full_ln_weight
        )
=======
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    key_start + 2,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 3,
                ]
            ]
            * full_ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_retained_start = self.key_start - 4
        second_key_retained_start = self.second_key_row - 7
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : first_key_retained_start
                ],
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:3],
                key_weight[3:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
            dim=0,
        )
=======
        second_key_retained_start = self.key_start - 4
        second_key_retained_end = second_key_retained_start + 1
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : second_key_retained_start
                ],
                key_weight[: self.head_dim],
                key_weight[self.head_dim : self.head_dim + 2],
                self.weight[
                    second_key_retained_start : second_key_retained_end
                ],
                key_weight[self.head_dim + 2 :],
                self.weight[second_key_retained_end:],
                value_weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactSharedProjection(nn.Module):
    """Projection with a zero-mean effective offset and retained value scalar."""
=======
class CompactSharedProjection(nn.Module):
    """Projection with one column and its effective offset gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = linear.weight
        self.shared_bias = shared_bias

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)
=======
        self.shared_bias = shared_bias

        width = linear.out_features
        column_basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            column_basis[: column + 1, column] = 1.0 / denom
            column_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("column_basis", column_basis, persistent=False)

        centered_column = (
            linear.weight[:, :1]
            - linear.weight[:, :1].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (column_basis.transpose(0, 1) @ centered_column).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 1:].detach().clone())

        basis = linear.weight.new_zeros(width, width - 2)
        for column in range(width - 2):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = self.weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, self.weight, full_bias)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_column = self.column_basis @ self.column_weight
        full_weight = torch.cat((compact_column, self.weight), dim=1)
        raw_bias = self.bias_basis @ self.bias
        raw_bias = torch.cat(
            (
                raw_bias[:-1],
                raw_bias[-1:] + self.shared_bias,
            )
        )
        value_offset = full_weight[:, -1] * self.shared_bias
        full_bias = raw_bias - (raw_bias + value_offset).mean()
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with two weight-column and bias uniform directions fixed."""
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with four weight-column and bias uniform directions fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_columns = (
            linear.weight[:, :2]
            - linear.weight[:, :2].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 2:].detach().clone())
=======
        selected_columns = torch.cat(
            (linear.weight[:, :3], linear.weight[:, -1:]),
            dim=1,
        )
        centered_columns = (
            selected_columns
            - selected_columns.mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 3:-1].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat((compact_columns, self.weight), dim=1)
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat(
            (
                compact_columns[:, :3],
                self.weight,
                compact_columns[:, 3:],
            ),
            dim=1,
        )
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
=======
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
>>>>>>> REPLACE