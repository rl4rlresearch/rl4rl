MECHANISM: Parameter-free singleton positional-origin gauge

HYPOTHESIS: Reproducing the qualified 1,596-parameter design and fixing the seventh token/position translation without creating a zero-length parameter will yield 1,595 parameters with at least 99% accuracy.

INTENDED_EDIT: Add the qualified two-value-row and third positional-row compactions, fix seven translation coordinates, and reconstruct the remaining centered one-coordinate positional origin as a constant zero tensor.

EVIDENCE: Six fixed translations with three centered positional rows and two compact value rows achieved 99.10% accuracy at 1,596 parameters; the seventh translation is the closest exact gauge extension, while its prior implementation produced no accuracy result.

<<<<<<< SEARCH
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
=======
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        origin_tail = self.position_weight.new_zeros(
            self.embedding_dim - self.fixed_coordinates
        )
        origin = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and four query-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
=======
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 : -2],
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
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        first_key_retained_start = self.key_start - 4
=======
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / self.ln_weight
        first_key_retained_start = self.key_start - 4
>>>>>>> REPLACE

<<<<<<< SEARCH
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
            ),
=======
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix five token/position translations, then center the remaining
        # positional-origin tail and positional rows 1 and 2.
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
        # gauges, compact three independently biased second-head query rows and
        # one shared-bias first-head query row, and quotient one MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
>>>>>>> REPLACE