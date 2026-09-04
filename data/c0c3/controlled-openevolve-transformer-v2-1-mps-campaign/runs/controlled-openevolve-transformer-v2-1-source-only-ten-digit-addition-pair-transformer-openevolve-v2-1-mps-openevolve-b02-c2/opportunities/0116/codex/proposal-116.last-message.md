MECHANISM: Nonadjacent attention-input LayerNorm scale gauge quotient

HYPOTHESIS: Reproducing the verified 1,586-parameter design and fixing both the first and last `ln1` scale coordinates will yield 1,585 parameters with at least 99% accuracy, because QKV absorbs both scales and the nonadjacent layout preserves a trainable bias on the leading coordinate.

INTENDED_EDIT: Zero-center positional rows 1–4, reconstruct first-and-last anchored scales for both LayerNorm branches, and propagate each reconstructed scale through compact QKV or `fc1`.

EVIDENCE: The single trailing `ln1` anchor achieved 99.92% accuracy at 1,586 parameters; analogously, nonadjacent first-and-last `ln2` anchors achieved 99.99%, while adjacent trailing anchors achieved only 98.79%.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and four row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and five row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:4] - embedding.weight[1:4].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[4:].detach().clone())
=======
        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(
        self,
        linear: nn.Linear,
        key_start: int,
        second_head_start: int,
        ln_weight: nn.Parameter,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight
=======
    def __init__(
        self,
        linear: nn.Linear,
        key_start: int,
        second_head_start: int,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        scaled_query_weight = (
            linear.weight[
                [
                    0,
                    self.head_dim,
                    self.head_dim + 1,
                    self.head_dim + 2,
                ]
            ]
            * ln_weight
        )
        centered_query_weight = (
            scaled_query_weight
            - scaled_query_weight.mean(dim=1, keepdim=True)
        )
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        scaled_value_weight = linear.weight[-2:] * ln_weight
        centered_value_weight = (
            scaled_value_weight
            - scaled_value_weight.mean(dim=1, keepdim=True)
        )
        self.value_weight = nn.Parameter(
            (centered_value_weight @ basis).detach().clone()
        )
=======
        full_ln_weight = self._full_ln_weight()
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * full_ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        scaled_query_weight = (
            linear.weight[
                [
                    0,
                    self.head_dim,
                    self.head_dim + 1,
                    self.head_dim + 2,
                ]
            ]
            * full_ln_weight
        )
        centered_query_weight = (
            scaled_query_weight
            - scaled_query_weight.mean(dim=1, keepdim=True)
        )
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        scaled_value_weight = linear.weight[-2:] * full_ln_weight
        centered_value_weight = (
            scaled_value_weight
            - scaled_value_weight.mean(dim=1, keepdim=True)
        )
        self.value_weight = nn.Parameter(
            (centered_value_weight @ basis).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / self.ln_weight
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / self.ln_weight
        first_key_retained_start = self.key_start - 4
=======
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
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
        full_ln_weight = self._full_ln_weight()
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / full_ln_weight
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / full_ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / full_ln_weight
        first_key_retained_start = self.key_start - 4
>>>>>>> REPLACE

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
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
        )
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
            fixed_weight_coordinates=2,
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 3.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
>>>>>>> REPLACE

<<<<<<< SEARCH
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
            )
=======
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
                block.ln1.fixed_weight_coordinates,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )
            block.mlp.fc1 = CompactFirstLinearRow(
                block.mlp.fc1,
                block.ln2.weight,
                block.ln2.fixed_weight_coordinates,
            )
>>>>>>> REPLACE