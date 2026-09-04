MECHANISM: Incremental second-head query-key rotational gauge fixing

HYPOTHESIS: Adding a second sequential rotation within the second attention head will reduce the model to 1533 parameters while retaining at least 99% accuracy, because it removes another exact query-key rotational gauge without changing initialized attention scores.

INTENDED_EDIT: Rotate second-head query/key channels 1–2 after its existing rotation, fix the resulting query coefficient to zero, and omit that coefficient from the trainable QKV parameterization.

EVIDENCE: The current four-rotation design achieved 99.26% at 1534 parameters, and each preceding query-key rotation qualified; applying the same established symmetry to the less-constrained second head is better supported than another mean-zero QKV-row constraint, whose second application achieved only 96.45%.

<<<<<<< SEARCH
class FourRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and four query-key rotations fixed."""
=======
class FiveRotationGaugeFixedQKV(nn.Module):
    """QKV map with one input-null and five query-key rotations fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("four-rotation gauge fixing requires two suitable heads")
=======
        if head_dim < 4 or 2 * head_dim > in_features:
            raise ValueError("five-rotation gauge fixing requires two suitable heads")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 1:-1]
        )
=======
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 1:]
        )
        self.head_two_second_weight = nn.Parameter(
            fixed_weight[self.second_query + 1, 1:]
        )
        self.trailing_weight = nn.Parameter(
            fixed_weight[self.second_query + 2:-1]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for query_start in (0, 1, 2, self.second_query):
=======
        for query_start in (0, 1, 2, self.second_query, self.second_query + 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 1:-1]
            )
=======
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 1:]
            )
            self.head_two_second_weight.copy_(
                fixed_weight[self.second_query + 1, 1:]
            )
            self.trailing_weight.copy_(
                fixed_weight[self.second_query + 2:-1]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        last_row = self.basis @ self.last_weight
=======
        head_two_row = F.pad(self.head_two_weight, (1, 0))
        head_two_second_row = F.pad(self.head_two_second_weight, (1, 0))
        last_row = self.basis @ self.last_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                self.trailing_weight,
=======
                self.first_head_tail,
                head_two_row.unsqueeze(0),
                head_two_second_row.unsqueeze(0),
                self.trailing_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = FourRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
=======
        self.qkv = FiveRotationGaugeFixedQKV(
            d_model, 3 * d_model, self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, FourRotationGaugeFixedQKV):
=======
        elif isinstance(module, FiveRotationGaugeFixedQKV):
>>>>>>> REPLACE