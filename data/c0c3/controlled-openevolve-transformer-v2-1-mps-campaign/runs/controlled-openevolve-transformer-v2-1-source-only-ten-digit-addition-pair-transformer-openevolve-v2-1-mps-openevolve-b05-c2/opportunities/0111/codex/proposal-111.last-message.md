MECHANISM: Twelfth within-head query–key rotation gauge fixing

HYPOTHESIS: Fixing the remaining second-head query-weight coordinate through a simultaneous query–key rotation will reduce the qualified model from 1495 to 1494 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add the unused `(second_query, 2)` rotation and reconstruct that query row with three leading zeros instead of two.

EVIDENCE: The qualified 1495-parameter model already preserves accuracy with eleven query–key rotations; unlike the unsuccessful final-LayerNorm bias anchors, this change extends the same initialization-preserving attention symmetry by one coordinate.

<<<<<<< SEARCH
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
=======
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query + 1, 1),
            (0, 2),
        )
=======
            (self.second_query + 1, 1),
            (0, 2),
            (self.second_query, 2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
=======
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE