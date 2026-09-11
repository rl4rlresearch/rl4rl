MECHANISM: Pre-attention LayerNorm-bias absorption gauge

HYPOTHESIS: Starting from the verified 1,607-parameter design, fixing one `ln1.bias` coordinate will yield 1,606 parameters and at least 99% accuracy because its constant downstream contribution is absorbable by the learned attention query/value and projection offsets.

INTENDED_EDIT: Reproduce the qualified positional-row and three-coordinate `ln2.bias` compactions, then reconstruct the final `ln1.bias` coordinate as zero.

EVIDENCE: Fixing three `ln2.bias` coordinates achieved 99.42% at 1,607 parameters, demonstrating that LayerNorm offsets followed by learned affine projections are redundant; the failed fourth `ln2` removal motivates applying the same redundancy independently at `ln1`.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and one origin-shift gauge fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and three row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("origin_basis", basis, persistent=False)

        origin_tail = embedding.weight[0, self.fixed_coordinates :]
=======
        self.register_buffer("origin_basis", basis, persistent=False)

        position_basis = embedding.weight.new_zeros(
            self.embedding_dim,
            self.embedding_dim - 1,
        )
        for column in range(self.embedding_dim - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            position_basis[: column + 1, column] = 1.0 / denom
            position_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("position_basis", position_basis, persistent=False)

        origin_tail = embedding.weight[0, self.fixed_coordinates :]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.origin_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_origin).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[1:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        full_weight = torch.cat((origin.unsqueeze(0), self.weight), dim=0)
        return F.embedding(idx, full_weight)
=======
        self.origin_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_origin).detach().clone()
        )

        centered_positions = embedding.weight[1:3] - embedding.weight[1:3].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[3:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        compact_positions = (
            self.position_weight @ self.position_basis.transpose(0, 1)
        )
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                compact_positions,
                self.weight,
            ),
            dim=0,
        )
        return F.embedding(idx, full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=1,
        )
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five token/position translations, then center the remaining
        # positional-origin tail using its residual-uniform shift gauge.
=======
        # Fix five token/position translations, then center the remaining
        # positional-origin tail and positional rows 1 and 2.
>>>>>>> REPLACE