MECHANISM: Nonadjacent fourth MLP residual-output uniform gauge quotient

HYPOTHESIS: Reproducing the verified seven-centered-key, three-centered-`fc2` design and additionally centering the final, nonadjacent `fc2` input column will yield 1,580 parameters with at least 99% accuracy, because its uniform output component is removed by the final LayerNorm while avoiding the adjacent-column layout that previously failed.

INTENDED_EDIT: Center seven qualified QKV key rows, retain only the remaining second-head key row, and represent `fc2` columns 0–2 plus its final column in the zero-mean output basis while reconstructing their original order.

EVIDENCE: The seven-key, three-column design achieved 99.87% accuracy at 1,581 parameters, whereas extending to the adjacent fourth `fc2` column achieved only 92.13%; earlier nonadjacent LayerNorm anchors succeeded where adjacent extensions failed, motivating the same separated-coordinate test for `fc2`.

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
class CompactResidualLinear(nn.Module):
    """Linear layer with two weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_columns = (
            linear.weight[:, :2]
            - linear.weight[:, :2].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 2:].detach().clone())

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat((compact_columns, self.weight), dim=1)
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with four weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

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

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

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