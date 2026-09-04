MECHANISM: Bias-decoupled second LayerNorm–MLP row gauge quotient

HYPOTHESIS: Reproducing the verified 1,609-parameter positional design and quotienting a second `fc1` row with dynamic LayerNorm-bias compensation will produce a 1,608-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact positional rows 0–2 as in the qualified 1,609-parameter design, then represent the first two `fc1` rows with seven-dimensional Helmert coordinates while decoupling the second row’s effective bias from `ln2.bias`.

EVIDENCE: Positional-row compaction achieved 99.36% at 1,609 parameters. The earlier second `fc1` quotient narrowly missed at 98.38% while using a bias coupled to the changing LayerNorm offset; explicitly compensating that offset tests the same exact gauge with better-conditioned optimization.

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
    """Position embedding with five translations and three row-shift gauges fixed."""

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
    """Linear layer with two LayerNorm input-direction gauges fixed."""

    def __init__(
        self,
        linear: nn.Linear,
        ln_weight: nn.Parameter,
        ln_bias: nn.Parameter,
    ):
        super().__init__()
        self.ln_weight = ln_weight
        self.ln_bias = ln_bias
        self.weight = nn.Parameter(linear.weight[2:].detach().clone())
        self.bias = linear.bias

        width = linear.in_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("row_basis", basis, persistent=False)

        scaled_rows = linear.weight[:2] * ln_weight
        centered_rows = scaled_rows - scaled_rows.mean(dim=1, keepdim=True)
        self.row_weight = nn.Parameter(
            (centered_rows @ basis).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_rows = self.row_weight @ self.row_basis.transpose(0, 1)
        first_rows = scaled_rows / self.ln_weight
        full_weight = torch.cat((first_rows, self.weight), dim=0)

        ln_bias = torch.cat((self.ln_bias, self.ln_bias.new_zeros(1)))
        second_bias = self.bias[1:2] - first_rows[1:2] @ ln_bias
        full_bias = torch.cat((self.bias[:1], second_bias, self.bias[2:]))
        return F.linear(x, full_weight, full_bias)
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
        # Fix five token/position translations, center the remaining
        # positional-origin tail, and quotient positional rows 1 and 2.
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
                block.ln2.bias,
            )
>>>>>>> REPLACE