MECHANISM: Tenth noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, 9, and 10 will produce a 1,600-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the current gauge set through the verified rows 8 and 9, then omit and reconstruct the final input coefficient of row 10 using the existing gauge-equivalent reparameterization.

EVIDENCE: The nine-row design achieved 99.75% accuracy with 1,601 parameters; adding row 10 is the smallest incremental reduction that continues to avoid row 3, whose inclusion previously reduced accuracy to 97.77%.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7)
=======
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10)
>>>>>>> REPLACE