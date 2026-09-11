MECHANISM: Per-coordinate query–key scaling gauge quotient

HYPOTHESIS: Applying an exact fixed-norm quotient to one zero-bias head-0 query row in the qualified shared-bias design will produce a 1,623-parameter model with at least 99% accuracy.

INTENDED_EDIT: Adopt the verified 1,624-parameter head-0 query-bias sharing layout, then represent one omitted-bias query row by seven stereographic direction parameters and inversely rescale its paired key row.

EVIDENCE: Head-0 bias sharing achieved 99.64% at 1,624 parameters, while both tested head-1 sharing reductions failed; the query–key rescaling symmetry removes a parameter without tying or deleting another sensitive query pathway and preserves initialized attention scores exactly.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection omitting one query bias, all key biases, and 4:3 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query_end = self.key_start - 1
        full_bias = torch.cat(
            (
                self.bias[:query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """Shared head-0 query bias with one query-key scaling gauge fixed."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.gauge_query = self.head_dim - 2
        self.gauge_key = self.key_start + self.gauge_query
        self.query_radius = 0.05

        query_row = linear.weight[self.gauge_query].detach()
        query_norm = query_row.norm()
        query_unit = query_row / query_norm
        stereographic = query_unit[:-1] / (1.0 + query_unit[-1])
        self.query_direction = nn.Parameter(stereographic.clone())

        transformed_weight = linear.weight.detach().clone()
        transformed_weight[self.gauge_key].mul_(query_norm / self.query_radius)
        self.weight = nn.Parameter(
            torch.cat(
                (
                    transformed_weight[: self.gauge_query],
                    transformed_weight[self.gauge_query + 1 :],
                )
            )
        )

        compact_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction_sq = self.query_direction.square().sum()
        query_unit = torch.cat(
            (
                2.0 * self.query_direction,
                (1.0 - direction_sq).unsqueeze(0),
            )
        ) / (1.0 + direction_sq)
        query_row = self.query_radius * query_unit
        full_weight = torch.cat(
            (
                self.weight[: self.gauge_query],
                query_row.unsqueeze(0),
                self.weight[self.gauge_query :],
            )
        )

        second_query_end = 1 + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:1].expand(self.head_dim - 2),
                self.bias.new_zeros(2),
                self.bias[1:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing one
        # head-1 query bias, all key biases, and value biases in a 4:3 layout.
=======
        # Use the qualified shared head-0 query-bias layout, then quotient one
        # zero-bias query row's exact inverse-scaling symmetry with its key row.
>>>>>>> REPLACE