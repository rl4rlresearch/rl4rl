MECHANISM: Third bias-absorbed LayerNorm query-row gauge quotient

HYPOTHESIS: Helmert-parameterizing the third independently biased second-head query row will reduce the verified 1,602-parameter design to 1,601 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct all three independently biased second-head query rows from seven-dimensional zero-mean coordinates.

EVIDENCE: The first and second query-row quotients achieved 99.93% at 1,603 parameters and 99.90% at 1,602 parameters, respectively, while extending key compaction failed; the remaining second-head query row has the same independent learned-bias structure.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and three biased query-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[: self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 :],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
=======
        self.key_weight = nn.Parameter(
            (centered_key_weight @ basis).detach().clone()
        )

        scaled_query_weight = (
            linear.weight[self.head_dim : self.head_dim + 3] * ln_weight
        )
        centered_query_weight = (
            scaled_query_weight
            - scaled_query_weight.mean(dim=1, keepdim=True)
        )
        self.query_weight = nn.Parameter(
            (centered_query_weight @ basis).detach().clone()
        )

        query_bias = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        second_retained_start = self.second_key_row - 2
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:2],
                self.weight[self.key_start : second_retained_start],
                key_weight[2:],
                self.weight[second_retained_start:],
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
        first_key_retained_start = self.key_start - 3
        second_key_retained_start = self.second_key_row - 5
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges, and quotient one independently biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus three biased query-row gauges, and quotient one
        # independently biased MLP input row.
>>>>>>> REPLACE