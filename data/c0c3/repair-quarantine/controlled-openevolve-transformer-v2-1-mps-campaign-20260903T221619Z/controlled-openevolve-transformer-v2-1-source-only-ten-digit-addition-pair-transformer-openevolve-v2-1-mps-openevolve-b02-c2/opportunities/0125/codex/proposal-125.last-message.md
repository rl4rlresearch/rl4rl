MECHANISM: Third nonadjacent MLP input-row LayerNorm-null gauge

HYPOTHESIS: Reproducing the verified 1,580-parameter design and centering the middle `fc1` row will yield 1,579 parameters with at least 99% accuracy, because its LayerNorm-null input component can be absorbed by that row’s independent bias.

INTENDED_EDIT: Apply the qualified seven-key, three-`fc2`-column, and first/last-`fc1`-row compactions, then represent the middle `fc1` row in the same seven-dimensional centered basis.

EVIDENCE: The first-and-last `fc1` row design achieved 99.82% accuracy at 1,580 parameters; a separated middle row has the same independently biased LayerNorm-null gauge, while the alternative fourth-`fc2`-column reduction reached only 97.44%.

<<<<<<< SEARCH
    """Compact QKV with five key, four query, and two value-row gauges fixed."""
=======
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
    """Linear layer with one LayerNorm input-direction gauge fixed."""
=======
    """Linear layer with three nonadjacent LayerNorm input gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
        self.bias = linear.bias
=======
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.middle_row = linear.out_features // 2
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.middle_row],
                linear.weight[self.middle_row + 1 : -1],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())
        self.bias = linear.bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_row = linear.weight[0] * self._full_ln_weight()
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )
=======
        scaled_rows = (
            linear.weight[[0, self.middle_row, -1]]
            * self._full_ln_weight()
        )
        centered_rows = scaled_rows - scaled_rows.mean(dim=1, keepdim=True)
        self.row_weight = nn.Parameter(
            (centered_rows @ basis).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self._full_ln_weight()
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_rows = self.row_weight @ self.row_basis.transpose(0, 1)
        selected_rows = scaled_rows / self._full_ln_weight()
        middle_retained_start = self.middle_row - 1
        full_weight = torch.cat(
            (
                selected_rows[:1],
                self.weight[:middle_retained_start],
                selected_rows[1:2],
                self.weight[middle_retained_start:],
                selected_rows[2:],
            ),
            dim=0,
        )
        return F.linear(x, full_weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    """Linear layer with two weight-column and bias uniform directions fixed."""
=======
    """Linear layer with three weight-column and bias uniform directions fixed."""
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
        centered_columns = (
            linear.weight[:, :3]
            - linear.weight[:, :3].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 3:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
=======
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient three nonadjacent independently biased MLP input rows.
>>>>>>> REPLACE