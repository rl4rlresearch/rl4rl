MECHANISM: Second downstream-projection-absorbed value-row LayerNorm gauge

HYPOTHESIS: Helmert-parameterizing the penultimate value row will reduce the verified model from 1,599 to 1,598 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component produces only a position-independent attention output that the learned projection-offset subspace can absorb.

INTENDED_EDIT: Extend `CompactQKV` from one to two compact value rows, reconstructing both final value rows from independent seven-dimensional zero-mean coordinates.

EVIDENCE: Compacting the final value row achieved 99.97% accuracy at 1,599 parameters; applying the same value-row quotient to the adjacent row is the closest supported reduction, while fifth-key and additional LayerNorm-bias reductions failed.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and one value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[second_head_start + 2 : -1],
=======
                linear.weight[second_head_start + 2 : -2],
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_value_weight = linear.weight[-1] * ln_weight
        centered_value_weight = (
            scaled_value_weight - scaled_value_weight.mean()
        )
        self.value_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_value_weight).detach().clone()
        )
=======
        scaled_value_weight = linear.weight[-2:] * ln_weight
        centered_value_weight = (
            scaled_value_weight
            - scaled_value_weight.mean(dim=1, keepdim=True)
        )
        self.value_weight = nn.Parameter(
            (centered_value_weight @ basis).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_value_weight = self.key_basis @ self.value_weight
=======
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                value_weight.unsqueeze(0),
=======
                value_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix four key rows, the four qualified query rows, and one value row;
        # also quotient one independently biased MLP input row.
=======
        # Fix four key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
>>>>>>> REPLACE