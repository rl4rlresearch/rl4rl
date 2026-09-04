MECHANISM: Zero-bias attention Q/K rotation gauge

HYPOTHESIS: Gauge-fixing one coordinate of the first head’s zero-bias query-row pair, with the matching orthogonal key-row rotation, will reduce the verified 1,593-parameter model to 1,592 parameters while retaining at least 99% accuracy because it preserves the initialized attention score matrix exactly.

INTENDED_EDIT: Givens-rotate query rows 2–3 and matching key rows 2–3, omit the resulting zero query coordinate, and reconstruct that coordinate as zero during forward passes.

EVIDENCE: The current two-column `fc2` quotient achieved 99.96% at 1,593 parameters, while a third `fc2` column and an attention-projection column both failed; an independent internal Q/K factorization symmetry is therefore the most informative next reduction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with row gauges and one zero-bias Q/K rotation fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = nn.Parameter(retained_weight.detach().clone())

        width = linear.in_features
=======
        # Query rows 2 and 3 and their matching key rows both have fixed zero
        # biases. Rotate each pair together, preserving QK^T, so one query
        # coordinate can be fixed to zero.
        query_pair = retained_weight[1:3]
        key_pair = retained_weight[4:6]
        radius = torch.sqrt(
            query_pair[0, 0].square() + query_pair[1, 0].square()
        )
        cosine = query_pair[1, 0] / radius
        sine = query_pair[0, 0] / radius
        rotated_query = torch.stack(
            (
                cosine * query_pair[0] - sine * query_pair[1],
                sine * query_pair[0] + cosine * query_pair[1],
            )
        )
        rotated_key = torch.stack(
            (
                cosine * key_pair[0] - sine * key_pair[1],
                sine * key_pair[0] + cosine * key_pair[1],
            )
        )
        retained_weight = torch.cat(
            (
                retained_weight[:1],
                rotated_query,
                retained_weight[3:4],
                rotated_key,
                retained_weight[6:],
            ),
            dim=0,
        )

        width = linear.in_features
        flat_weight = retained_weight.flatten()
        self.weight = nn.Parameter(
            torch.cat(
                (
                    flat_weight[:width],
                    flat_weight[width + 1 :],
                )
            ).detach().clone()
        )

>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        value_weight = scaled_value_weight / self.ln_weight
        retained_flat = torch.cat(
            (
                self.weight[: self.key_start],
                self.weight.new_zeros(1),
                self.weight[self.key_start :],
            )
        )
        retained_weight = retained_flat.view(-1, self.key_start)
        first_key_retained_start = self.key_start - 4
        second_key_retained_start = self.second_key_row - 6
        full_weight = torch.cat(
            (
                query_weight[:1],
                retained_weight[: self.head_dim - 1],
                query_weight[1:],
                retained_weight[
                    self.head_dim - 1 : first_key_retained_start
                ],
                key_weight[:2],
                retained_weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:],
                retained_weight[second_key_retained_start:],
                value_weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE