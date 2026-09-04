MECHANISM: Second independently biased MLP LayerNorm-row gauge quotient

HYPOTHESIS: Reproducing the verified 1,595-parameter design and Helmert-parameterizing a second `fc1` row will yield 1,594 parameters with at least 99% accuracy, because each row’s independent bias can absorb the removed LayerNorm-uniform weight component.

INTENDED_EDIT: Apply the qualified seven-translation, three-position-row, four-query-row, and two-value-row compactions, then reconstruct the first two MLP input rows from independent seven-dimensional zero-mean coordinates.

EVIDENCE: The 1,595-parameter reference achieved 99.97% accuracy while already applying this quotient to one independently biased `fc1` row; extending the same local parameterization to the adjacent independently biased row is more directly supported than the failed third-value-row or fourth-position-row reductions.

<<<<<<< SEARCH
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
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and four row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 7

        position_basis = embedding.weight.new_zeros(
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        centered_positions = embedding.weight[1:4] - embedding.weight[1:4].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[4:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.origin_basis @ self.origin_weight
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.position_weight.new_zeros(
            self.embedding_dim - self.fixed_coordinates
        )
        origin = torch.cat(
            (
                origin_tail.new_zeros(self.fixed_coordinates),
                origin_tail,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and two biased query-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[: self.head_dim],
                linear.weight[self.head_dim + 2 : key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 : -2],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_query_weight = (
            linear.weight[self.head_dim : self.head_dim + 2] * ln_weight
        )
        centered_query_weight = (
            scaled_query_weight
            - scaled_query_weight.mean(dim=1, keepdim=True)
        )
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
=======
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

        query_bias = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / self.ln_weight
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        first_key_retained_start = self.key_start - 2
        second_key_retained_start = self.second_key_row - 4
        full_weight = torch.cat(
            (
                self.weight[: self.head_dim],
                query_weight,
                self.weight[
                    self.head_dim : first_key_retained_start
                ],
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
            ),
            dim=0,
        )
=======
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
        second_key_retained_start = self.second_key_row - 6
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
                key_weight[2:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
            dim=0,
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
    """Linear layer with two LayerNorm input-direction gauges fixed."""

    def __init__(self, linear: nn.Linear, ln_weight: nn.Parameter):
        super().__init__()
        self.ln_weight = ln_weight
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
        return F.linear(x, full_weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 3.
        with torch.no_grad():
            for coordinate in range(7):
                offset = self.pos_emb.weight[0, coordinate].detach().clone()
                self.token_emb.weight[:, coordinate].add_(offset)
                self.pos_emb.weight[:, coordinate].sub_(offset)

            origin_shift = -self.pos_emb.weight[0, 7:].mean()
            self.token_emb.weight[:, :7].add_(origin_shift)
            self.pos_emb.weight[:, :7].sub_(origin_shift)
            self.pos_emb.weight[0].add_(origin_shift)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus two biased query-row gauges, and quotient one independently
        # biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key rows,
        # four qualified query rows, and two value rows; also quotient two
        # independently biased MLP input rows.
>>>>>>> REPLACE