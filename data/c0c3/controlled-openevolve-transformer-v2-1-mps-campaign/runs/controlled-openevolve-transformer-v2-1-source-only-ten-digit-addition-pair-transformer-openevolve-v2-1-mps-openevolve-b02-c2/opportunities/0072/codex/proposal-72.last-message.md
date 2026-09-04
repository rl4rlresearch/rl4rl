MECHANISM: Per-position residual common-mode quotient

HYPOTHESIS: Extending the verified 1,611-parameter design by zero-centering positional row 1 will yield 1,610 parameters and at least 99% accuracy, because that row’s removed uniform component survives only as a position-local residual shift canceled by every downstream LayerNorm.

INTENDED_EDIT: Reproduce the verified five translation gauges, positional-origin quotient, and compact `fc2` bias, then represent positional row 1 with seven Helmert coordinates.

EVIDENCE: The positional-origin common-mode quotient achieved 99.92% at 1,611 parameters; applying the same residual-uniform invariance to a separate full positional row is directly supported and avoids the failed sixth coordinate-wise translation.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Positional embedding with two token-position translation gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.weight = nn.Parameter(embedding.weight.detach().reshape(-1)[2:].clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_zeros(2), self.weight))
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

        centered_position = embedding.weight[1] - embedding.weight[1].mean()
        self.position_weight = nn.Parameter(
            (position_basis.transpose(0, 1) @ centered_position).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[2:].detach().clone())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
        second_position = self.position_basis @ self.position_weight
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                second_position.unsqueeze(0),
                self.weight,
            ),
            dim=0,
        )
        return F.embedding(idx, full_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
=======
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
    def __init__(self, d_model: int, d_ff: int, dropout: float):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix two exact token/position translation gauges while preserving the
        # initialized hidden inputs and output probabilities.
        with torch.no_grad():
            for coordinate in range(2):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)
        self.pos_emb = CompactPositionEmbedding(self.pos_emb)
=======
        # Fix five token/position translations, then center the remaining
        # positional origin using its residual-uniform shift gauge. The compact
        # embedding also quotients the uniform direction of positional row 1.
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