MECHANISM: Second pre-MLP LayerNorm-bias absorption gauge

HYPOTHESIS: Extending the verified 1,609-parameter design to fix a second `ln2.bias` coordinate will yield 1,608 parameters and at least 99% accuracy, because both removed LayerNorm offsets are exactly absorbable by the retained full `fc1.bias`.

INTENDED_EDIT: Reproduce the qualified positional-row and `fc2.bias` compactions, then retain six rather than seven `ln2.bias` coordinates and reconstruct its final two coordinates as zeros.

EVIDENCE: The 1,609-parameter positional design achieved 99.36% accuracy while already fixing one `ln2.bias` coordinate. The failed 1,608 attempts instead constrained positional row 3 or a second `fc1` weight row, motivating this independent continuation of the already-qualified downstream-bias redundancy.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with five token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[5:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(5), self.weight))
        return F.embedding(
            idx,
            full_weight.view(self.num_embeddings, self.embedding_dim),
        )
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and two row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 5

        origin_width = self.embedding_dim - self.fixed_coordinates
        origin_basis = embedding.weight.new_zeros(origin_width, origin_width - 1)
        for column in range(origin_width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            origin_basis[: column + 1, column] = 1.0 / denom
            origin_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("origin_basis", origin_basis, persistent=False)

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
        centered_origin = origin_tail - origin_tail.mean()
        self.origin_weight = nn.Parameter(
            (origin_basis.transpose(0, 1) @ centered_origin).detach().clone()
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
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
=======
class CompactLayerNormBias(nn.Module):
    """LayerNorm with two downstream-linear-absorbed bias coordinates fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-2].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)


class MLP(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_row = self.row_basis @ self.row_weight
        first_row = scaled_row / self.ln_weight
        full_weight = torch.cat((first_row.unsqueeze(0), self.weight), dim=0)
        return F.linear(x, full_weight, self.bias)


class CompactResidualLinear(nn.Module):
    """Linear layer with its residual-uniform bias direction fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five exact token/position translation gauges while preserving
        # the initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(5):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
=======
        # Fix five token/position translations, then center the remaining
        # positional origin using its residual-uniform shift gauge. The compact
        # embedding also quotients positional rows 1 and 2.
        with torch.no_grad():
            for coordinate in range(5):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)

            origin_shift = -self.pos_emb.weight[0, 5:].mean()
            self.token_emb.weight[:, :5].add_(origin_shift)
            self.pos_emb.weight[:, :5].sub_(origin_shift)
            self.pos_emb.weight[0].add_(origin_shift)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
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
            )
            block.mlp.fc2 = CompactResidualLinear(block.mlp.fc2)
>>>>>>> REPLACE