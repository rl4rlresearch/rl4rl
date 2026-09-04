MECHANISM: Orthonormal LayerNorm–key-weight gauge quotient

HYPOTHESIS: Expressing one key-projection row in a seven-dimensional zero-mean basis scaled by `ln1.weight` will reduce the verified model to 1,620 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key shift and cannot change causal attention probabilities.

INTENDED_EDIT: Replace one eight-scalar QKV key-weight row with seven Helmert-basis coordinates, reconstructing it through the learned `ln1` scales while preserving all other QKV weights and compact biases.

EVIDENCE: The current orthonormal attention-offset quotient reached 99.57% at 1,621 parameters, while every qualified reference already removes all key biases without harming accuracy; together these results motivate another well-conditioned attention-invariant quotient targeting constant key offsets rather than a sensitive learned pathway.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection with shared head-0 queries and one retained value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
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
        return F.linear(x, self.weight, full_bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified query and value-bias layout. Express the
        # combined value/projection offset modulo the uniform residual
        # direction, using six orthonormal coordinates plus the value scalar.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
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
>>>>>>> REPLACE