MECHANISM: Orthonormal LayerNorm–MLP row gauge quotient

HYPOTHESIS: Starting from the verified 1,618-parameter design, expressing one `fc1` weight row in a seven-dimensional zero-mean basis scaled by `ln2.weight` will yield 1,617 parameters and at least 99% accuracy, because the removed direction affects only a constant preactivation offset that the row’s independent learned bias preserves.

INTENDED_EDIT: Reproduce the qualified three-key-row quotient, then quotient the first MLP input-weight row while retaining its full learned bias.

EVIDENCE: Three key-row quotients achieved 99.96% at 1,618 parameters, while alternative fourth key quotients reached only 91.96% and 98.25%; this motivates applying the successful orthonormal LayerNorm gauge to an MLP row with an independent bias instead of another sensitive key coordinate.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with one LayerNorm-induced key-weight gauge fixed."""

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
        self.key_row = key_start
        self.ln_weight = ln_weight

        retained_weight = torch.cat(
            (
                linear.weight[: self.key_row],
                linear.weight[self.key_row + 1 :],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("key_basis", basis, persistent=False)

        scaled_key_weight = linear.weight[self.key_row] * ln_weight
        centered_key_weight = scaled_key_weight - scaled_key_weight.mean()
        self.key_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_key_weight).detach().clone()
        )

        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_key_weight = self.key_basis @ self.key_weight
        key_weight = scaled_key_weight / self.ln_weight
        full_weight = torch.cat(
            (
                self.weight[: self.key_row],
                key_weight.unsqueeze(0),
                self.weight[self.key_row :],
            ),
            dim=0,
        )
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, full_weight, full_bias)
=======
class CompactQKV(nn.Module):
    """Compact QKV with three LayerNorm-induced key-weight gauges fixed."""

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

        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
        self.weight = nn.Parameter(retained_weight.detach().clone())

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("key_basis", basis, persistent=False)

        scaled_key_weight = (
            linear.weight[[key_start, key_start + 1, second_head_start]]
            * ln_weight
        )
        centered_key_weight = (
            scaled_key_weight - scaled_key_weight.mean(dim=1, keepdim=True)
        )
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
=======
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified query and value-bias layout. Quotient one key
        # row against LayerNorm's uniform normalized-input direction, then
        # express the combined value/projection offset modulo uniform residuals.
        for block in self.blocks:
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
=======
        # Retain the qualified bias and projection layout, fix three key-row
        # gauges, and quotient one independently biased MLP input row.
        for block in self.blocks:
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
>>>>>>> REPLACE