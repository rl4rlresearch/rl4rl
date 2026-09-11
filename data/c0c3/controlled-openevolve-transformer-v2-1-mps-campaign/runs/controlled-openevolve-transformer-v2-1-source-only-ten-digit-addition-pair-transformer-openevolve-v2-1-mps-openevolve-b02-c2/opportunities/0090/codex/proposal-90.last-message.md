MECHANISM: Third downstream-projection-absorbed value-row LayerNorm gauge

HYPOTHESIS: Helmert-parameterizing a third value row will reduce the qualified 1,598-parameter design to 1,597 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct the final three value rows from independent seven-dimensional zero-mean coordinates.

EVIDENCE: Compacting one value row achieved 99.97% accuracy at 1,599 parameters, and compacting the adjacent second row achieved 99.55% at 1,598; extending the same successful quotient to the next adjacent row is the closest supported reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and four query-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and three value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[second_head_start + 2 :],
=======
                linear.weight[second_head_start + 2 : -3],
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

        scaled_value_weight = linear.weight[-3:] * ln_weight
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
                key_weight[2:],
                self.weight[second_key_retained_start:],
            ),
=======
                key_weight[2:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges, compact three independently biased second-head query rows and
        # one shared-bias first-head query row, and quotient one MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key rows,
        # four query rows, and three value rows, and quotient one MLP input row.
>>>>>>> REPLACE