MECHANISM: Complete pre-MLP LayerNorm-bias absorption

HYPOTHESIS: Reproducing the verified 1,577-parameter design and fixing the remaining `ln2` bias coordinate will yield 1,576 parameters with at least 99% accuracy, because every `ln2` bias contribution can be represented by the learned downstream `fc1` bias.

INTENDED_EDIT: Apply the qualified positional, seven-key, and nonadjacent four-column `fc2` compactions, then remove the final trainable `ln2` bias coordinate without creating a zero-length parameter.

EVIDENCE: The combined 1,577-parameter design achieved 99.81% accuracy, while independently increasing `ln2` bias fixation from six to seven coordinates achieved 99.97%; the remaining coordinate has the same downstream-bias absorption mechanism.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and five row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and seven row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())
=======
        centered_positions = embedding.weight[1:7] - embedding.weight[1:7].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[7:].detach().clone())
>>>>>>> REPLACE

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
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )
=======
        if fixed_coordinates < layer_norm.bias.numel():
            self.bias = nn.Parameter(
                layer_norm.bias[:-fixed_coordinates].detach().clone()
            )
        else:
            self.register_parameter("bias", None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
=======
        if self.bias is None:
            full_bias = full_weight.new_zeros(self.fixed_coordinates)
        else:
            full_bias = torch.cat(
                (self.bias, self.bias.new_zeros(self.fixed_coordinates))
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
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=8,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 6.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
=======
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
>>>>>>> REPLACE