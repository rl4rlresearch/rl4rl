MECHANISM: LayerNorm-nullspace key-row quotient

HYPOTHESIS: Representing the first key row in the orthonormal mean-zero basis will reduce the qualified 1490-parameter model to 1489 parameters while retaining at least 99% accuracy, because pre-attention LayerNorm makes its discarded common-mode component functionally invisible.

INTENDED_EDIT: Split the existing query/key tensor around the first key row, store that row with seven learned basis coordinates, and reconstruct its eight-coordinate weight during every forward pass.

EVIDENCE: The current 1490-parameter design reached 99.92% accuracy, while several further query–key GL gauges timed out. The qualified model already uses this mean-zero input quotient successfully in the MLP and final QKV row, motivating an independent, well-conditioned reduction.

<<<<<<< SEARCH
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features]
        )
=======
        self.pre_key_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:self.in_features]
        )
        self.first_key_weight = nn.Parameter(
            fixed_weight[self.in_features] @ basis
        )
        self.post_key_weight = nn.Parameter(
            fixed_weight[self.in_features + 1:2 * self.in_features]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
            )
=======
            self.pre_key_weight.copy_(
                fixed_weight[self.second_query + 3:self.in_features]
            )
            self.first_key_weight.copy_(
                fixed_weight[self.in_features] @ self.basis
            )
            self.post_key_weight.copy_(
                fixed_weight[self.in_features + 1:2 * self.in_features]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_value_row = F.pad(self.first_value_weight, (2, 0))
=======
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_key_row = self.basis @ self.first_key_weight
        first_value_row = F.pad(self.first_value_weight, (2, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_third_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
=======
                head_two_third_row.unsqueeze(0),
                self.pre_key_weight,
                first_key_row.unsqueeze(0),
                self.post_key_weight,
                first_value_row.unsqueeze(0),
>>>>>>> REPLACE