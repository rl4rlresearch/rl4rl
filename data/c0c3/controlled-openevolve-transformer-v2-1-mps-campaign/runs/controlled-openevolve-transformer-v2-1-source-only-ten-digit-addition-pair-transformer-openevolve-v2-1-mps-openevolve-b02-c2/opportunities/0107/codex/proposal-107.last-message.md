MECHANISM: Downstream-linear LayerNorm scale gauge quotient

HYPOTHESIS: Fixing one `ln2` scale coordinate in the verified six-bias-coordinate design will produce 1,589 parameters while retaining at least 99% accuracy, because the corresponding trainable `fc1` column absorbs the removed scale without reducing the learned function class.

INTENDED_EDIT: Use the qualified six-coordinate `ln2` bias reduction, reconstruct one `ln2` scale coordinate as constant one, and make the compact first MLP row use the reconstructed full scale vector.

EVIDENCE: The six-coordinate `ln2` design reached 99.88% and 99.77% accuracy at 1,590 parameters, whereas fixing a seventh bias coordinate fell to 97.17%; this tests an independent exact downstream-affine redundancy instead of repeating that failed bias reduction.

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
            self.weight = nn.Parameter(
                layer_norm.weight[:-fixed_weight_coordinates].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
        self.bias = nn.Parameter(
            layer_norm.bias[:-fixed_coordinates].detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_weight_coordinates:
            full_weight = torch.cat(
                (
                    self.weight,
                    self.weight.new_ones(self.fixed_weight_coordinates),
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
    def __init__(self, linear: nn.Linear, ln_weight: nn.Parameter):
        super().__init__()
        self.ln_weight = ln_weight
        self.weight = nn.Parameter(linear.weight[1:].detach().clone())
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_row = linear.weight[0] * ln_weight
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
=======
        scaled_row = linear.weight[0] * self._full_ln_weight()
        centered_row = scaled_row - scaled_row.mean()
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            return torch.cat(
                (
                    self.ln_weight,
                    self.ln_weight.new_ones(
                        self.fixed_ln_weight_coordinates
                    ),
                )
            )
        return self.ln_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self._full_ln_weight()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=4,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=1,
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