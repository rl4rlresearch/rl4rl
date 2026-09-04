MECHANISM: Additional LayerNorm-invisible QKV row gauge fixing

HYPOTHESIS: Encoding one additional unconstrained QKV row in the mean-zero basis will reduce the qualified 1493-parameter model to 1492 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the fully parameterized final query row with seven learned mean-zero coordinates and reconstruct its eight-dimensional weight before attention.

EVIDENCE: The current design achieved 0.9994 accuracy with 1493 parameters while applying the identical mean-zero basis construction to the final QKV row; extending that exact LayerNorm input gauge avoids the unsuccessful tenth value/output rotation.

<<<<<<< SEARCH
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 3:2 * self.in_features]
        )
=======
        self.pre_value_first_weight = nn.Parameter(
            fixed_weight[self.second_query + 3] @ basis
        )
        self.pre_value_weight = nn.Parameter(
            fixed_weight[self.second_query + 4:2 * self.in_features]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 3:2 * self.in_features]
            )
=======
            self.pre_value_first_weight.copy_(
                fixed_weight[self.second_query + 3] @ self.basis
            )
            self.pre_value_weight.copy_(
                fixed_weight[self.second_query + 4:2 * self.in_features]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        first_value_row = F.pad(self.first_value_weight, (2, 0))
=======
        head_two_third_row = F.pad(self.head_two_third_weight, (1, 0))
        pre_value_first_row = self.basis @ self.pre_value_first_weight
        first_value_row = F.pad(self.first_value_weight, (2, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_two_third_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
=======
                head_two_third_row.unsqueeze(0),
                pre_value_first_row.unsqueeze(0),
                self.pre_value_weight,
                first_value_row.unsqueeze(0),
>>>>>>> REPLACE