MECHANISM: Sixth downstream-affine LayerNorm-bias gauge

HYPOTHESIS: Reproducing the verified 1,591-parameter design and fixing a sixth `ln2` bias coordinate will yield 1,590 parameters with at least 99% accuracy, because `fc1`’s independent biases can absorb that LayerNorm offset.

INTENDED_EDIT: Apply the qualified seven-translation, three-position-row, four-query-row, two-value-row, and two-`fc2`-column compactions, then increase `ln2`’s fixed bias coordinates from five to six.

EVIDENCE: Fixing the fifth `ln2` bias coordinate achieved 99.89% accuracy at 1,591 parameters; extending that successful downstream-affine redundancy is better supported than the failed third-`fc2` column, second-`fc1` row, attention-projection column, or Q/K rotation reductions.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and two row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 5
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and four row-shift gauges fixed."""

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.fixed_coordinates = 7
>>>>>>> REPLACE

<<<<<<< SEARCH
        origin_width = self.embedding_dim - self.fixed_coordinates
        origin_basis = embedding.weight.new_zeros(origin_width, origin_width - 1)
        for column in range(origin_width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            origin_basis[: column + 1, column] = 1.0 / denom
            origin_basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("origin_basis", origin_basis, persistent=False)

=======
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
        origin_tail = self.origin_basis @ self.origin_weight
=======
        origin_tail = self.position_weight.new_zeros(
            self.embedding_dim - self.fixed_coordinates
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
=======
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
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with two weight-column and bias uniform directions fixed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()

        width = linear.out_features
        basis = linear.weight.new_zeros(width, width - 1)
        for column in range(width - 1):
            denom = math.sqrt((column + 1) * (column + 2))
            basis[: column + 1, column] = 1.0 / denom
            basis[column + 1, column] = -(column + 1) / denom
        self.register_buffer("bias_basis", basis, persistent=False)

        centered_columns = (
            linear.weight[:, :2]
            - linear.weight[:, :2].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 2:].detach().clone())

        centered_bias = linear.bias - linear.bias.mean()
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ centered_bias).detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compact_columns = self.bias_basis @ self.column_weight
        full_weight = torch.cat((compact_columns, self.weight), dim=1)
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=3,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
        )
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
        # Fix four key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
>>>>>>> REPLACE