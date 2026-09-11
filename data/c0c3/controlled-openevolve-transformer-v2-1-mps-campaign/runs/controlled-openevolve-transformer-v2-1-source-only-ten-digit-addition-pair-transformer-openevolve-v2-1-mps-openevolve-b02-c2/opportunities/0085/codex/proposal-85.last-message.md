MECHANISM: Second bias-absorbed LayerNorm query-row gauge quotient

HYPOTHESIS: Helmert-parameterizing a second query row in the second attention head will reduce the verified model from 1,603 to 1,602 parameters while retaining at least 99% accuracy, because its removed LayerNorm-uniform component is absorbable by that row’s independent learned query bias.

INTENDED_EDIT: Extend `CompactQKV` to reconstruct the first two second-head query rows from independent seven-dimensional zero-mean coordinates while preserving their learned biases.

EVIDENCE: Compacting the first second-head query row achieved 99.93% accuracy at 1,603 parameters, whereas extending key-row compaction to a fifth row failed at 72.64%; repeating the successful query-row quotient is the strongest supported next reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and one biased query-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four key-row and two biased query-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[self.head_dim + 1 : key_start],
=======
                linear.weight[self.head_dim + 2 : key_start],
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_query_weight = linear.weight[self.head_dim] * ln_weight
        centered_query_weight = (
            scaled_query_weight - scaled_query_weight.mean()
        )
        self.query_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_query_weight).detach().clone()
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_query_weight = self.key_basis @ self.query_weight
        query_weight = scaled_query_weight / self.ln_weight
=======
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / self.ln_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_retained_start = self.key_start - 1
        second_key_retained_start = self.second_key_row - 3
        full_weight = torch.cat(
            (
                self.weight[: self.head_dim],
                query_weight.unsqueeze(0),
=======
        first_key_retained_start = self.key_start - 2
        second_key_retained_start = self.second_key_row - 4
        full_weight = torch.cat(
            (
                self.weight[: self.head_dim],
                query_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus one biased query-row gauge, and quotient one independently
        # biased MLP input row.
=======
        # Retain the qualified bias and projection layout, fix four key-row
        # gauges plus two biased query-row gauges, and quotient one independently
        # biased MLP input row.
>>>>>>> REPLACE