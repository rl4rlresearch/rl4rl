MECHANISM: Third common-output MLP projection quotient

HYPOTHESIS: Gauge-fixing `fc2.weight[4, 6]` will reduce the verified model from 1606 to 1605 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove row 4 from a third, central MLP output column; the existing reconstruction, virtual AdamW moments, quotient updates, and gradient clipping already support arbitrary numbers of omitted coordinates.

EVIDENCE: The current design achieved 99.98% accuracy after extending the proven MLP common-output quotient from column 11 to the distant column 0, directly supporting another independent column quotient; column 6 tests an interior hidden feature rather than the previously used endpoints.

<<<<<<< SEARCH
        self.fixed_weight_rows = (4, 4)
        self.fixed_weight_columns = (in_features - 1, 0)
=======
        self.fixed_weight_rows = (4, 4, 4)
        self.fixed_weight_columns = (in_features - 1, 0, in_features // 2)
>>>>>>> REPLACE