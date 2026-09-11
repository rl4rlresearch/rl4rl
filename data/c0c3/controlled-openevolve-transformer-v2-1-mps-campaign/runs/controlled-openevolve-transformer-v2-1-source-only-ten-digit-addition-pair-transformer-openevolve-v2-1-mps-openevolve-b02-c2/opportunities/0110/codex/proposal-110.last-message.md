MECHANISM: Third downstream LayerNorm scale gauge with trainable-bias leading anchor

HYPOTHESIS: Fixing `ln2` scale coordinates 0, 1, and 7 will reduce the qualified 1,588-parameter design to 1,587 parameters while retaining at least 99% accuracy, because `fc1` absorbs the scales and coordinate 1 retains an independent trainable LayerNorm bias.

INTENDED_EDIT: Reconstruct three `ln2` scale coordinates as constant one and use the reconstructed full scale in the compact first MLP row.

EVIDENCE: The nonadjacent coordinate-0-and-7 anchors achieved 99.99% accuracy at 1,588 parameters, whereas anchoring the two trailing, bias-fixed coordinates achieved only 98.79%; extending the leading side tests a third anchor whose bias remains trainable.

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm, fixed_coordinates: int):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.fixed_coordinates = fixed_coordinates
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with downstream-linear-absorbed affine coordinates fixed."""

    def __init__(
        self,
        layer_norm: nn.LayerNorm,
        fixed_coordinates: int,
        fixed_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.fixed_coordinates = fixed_coordinates
        self.fixed_weight_coordinates = fixed_weight_coordinates
        if fixed_weight_coordinates:
            leading_fixed = fixed_weight_coordinates - 1
            self.weight = nn.Parameter(
                layer_norm.weight[leading_fixed:-1].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_weight_coordinates:
            leading_fixed = self.fixed_weight_coordinates - 1
            full_weight = torch.cat(
                (
                    self.weight.new_ones(leading_fixed),
                    self.weight,
                    self.weight.new_ones(1),
                )
            )
        else:
            full_weight = self.weight
        full_bias = torch.cat(
            (self.bias, self.bias.new_zeros(self.fixed_coordinates))
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            full_bias,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactFirstLinearRow(nn.Module):
    """Linear layer with one LayerNorm input-direction gauge fixed."""

    def __init__(self, linear: nn.Linear, ln_weight: nn.Parameter):
        super().__init__()
        self.ln_weight = ln_weight
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_row = linear.weight[0] * ln_weight
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)
=======
class CompactFirstLinearRow(nn.Module):
    """Linear layer with one LayerNorm input-direction gauge fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_row = linear.weight[0] * self._full_ln_weight()
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            leading_fixed = self.fixed_ln_weight_coordinates - 1
            return torch.cat(
                (
                    self.ln_weight.new_ones(leading_fixed),
                    self.ln_weight,
                    self.ln_weight.new_ones(1),
                )
            )
        return self.ln_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self._full_ln_weight()
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )
=======
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
                block.ln2.fixed_weight_coordinates,
            )
>>>>>>> REPLACE