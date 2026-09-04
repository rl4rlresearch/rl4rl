MECHANISM: Fifth MLP input-row LayerNorm-null gauge quotient

HYPOTHESIS: Reproducing the verified 1,578-parameter four-row design and centering the separated third-quarter `fc1` row will yield 1,577 parameters with at least 99% accuracy, because its LayerNorm-null input component can be absorbed by that row’s independent bias.

INTENDED_EDIT: Represent `fc1` rows 0, 3, 6, 9, and 11 in the seven-dimensional centered basis and reconstruct them in their original order.

EVIDENCE: Centering rows 0, 3, 6, and 11 achieved 99.91% accuracy at 1,578 parameters; row 9 is another separated row with the identical independently biased LayerNorm-null gauge.

<<<<<<< SEARCH
class CompactFirstLinearRow(nn.Module):
    """Linear layer with two nonadjacent LayerNorm input gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.weight = nn.Parameter(linear.weight[1:-1].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_rows = linear.weight[[0, -1]] * self._full_ln_weight()
        centered_rows = scaled_rows - scaled_rows.mean(dim=1, keepdim=True)
        self.row_weight = nn.Parameter(
            (centered_rows @ basis).detach().clone()
        )
=======
class CompactFirstLinearRow(nn.Module):
    """Linear layer with five separated LayerNorm input gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.quarter_row = linear.out_features // 4
        self.middle_row = linear.out_features // 2
        self.three_quarter_row = 3 * linear.out_features // 4
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.quarter_row],
                linear.weight[self.quarter_row + 1 : self.middle_row],
                linear.weight[
                    self.middle_row + 1 : self.three_quarter_row
                ],
                linear.weight[self.three_quarter_row + 1 : -1],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_rows = (
            linear.weight[
                [
                    0,
                    self.quarter_row,
                    self.middle_row,
                    self.three_quarter_row,
                    -1,
                ]
            ]
            * self._full_ln_weight()
        )
        centered_rows = scaled_rows - scaled_rows.mean(dim=1, keepdim=True)
        self.row_weight = nn.Parameter(
            (centered_rows @ basis).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_rows = self.row_weight @ self.row_basis.transpose(0, 1)
        edge_rows = scaled_rows / self._full_ln_weight()
        full_weight = torch.cat(
            (edge_rows[:1], self.weight, edge_rows[1:]),
            dim=0,
        )
        return F.linear(x, full_weight, self.bias)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_rows = self.row_weight @ self.row_basis.transpose(0, 1)
        selected_rows = scaled_rows / self._full_ln_weight()
        quarter_retained_start = self.quarter_row - 1
        middle_retained_start = self.middle_row - 2
        three_quarter_retained_start = self.three_quarter_row - 3
        full_weight = torch.cat(
            (
                selected_rows[:1],
                self.weight[:quarter_retained_start],
                selected_rows[1:2],
                self.weight[
                    quarter_retained_start:middle_retained_start
                ],
                selected_rows[2:3],
                self.weight[
                    middle_retained_start:three_quarter_retained_start
                ],
                selected_rows[3:4],
                self.weight[three_quarter_retained_start:],
                selected_rows[4:],
            ),
            dim=0,
        )
        return F.linear(x, full_weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient two nonadjacent independently biased MLP input rows.
=======
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient five separated independently biased MLP input rows.
>>>>>>> REPLACE