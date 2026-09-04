MECHANISM: Residual-uniform positional-row gauge quotient

HYPOTHESIS: Reproducing the verified 1,597-parameter design and Helmert-parameterizing positional row 4 will yield 1,596 parameters with at least 99% accuracy, because its position-local uniform residual shift is removed by downstream LayerNorms.

INTENDED_EDIT: Apply the qualified four-query/two-value QKV compaction and center positional rows 1–4 instead of rows 1–2.

EVIDENCE: Centering positional row 3 produced 99.82% accuracy at 1,597 parameters, while compacting a third value row fell to 98.92%; extending the successful exact positional gauge is the strongest next reduction.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and two row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with five translations and five row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:3] - embedding.weight[1:3].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[3:].detach().clone())
=======
        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())
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
        # Fix five token/position translations, then center the remaining
        # positional origin using its residual-uniform shift gauge. The compact
        # embedding also quotients the uniform direction of positional row 1.
=======
        # Fix five token/position translations, then center the remaining
        # positional origin using its residual-uniform shift gauge. The compact
        # embedding also quotients the uniform direction of positional rows 1–4.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus two biased query-row gauges, and quotient one independently
        # biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
>>>>>>> REPLACE