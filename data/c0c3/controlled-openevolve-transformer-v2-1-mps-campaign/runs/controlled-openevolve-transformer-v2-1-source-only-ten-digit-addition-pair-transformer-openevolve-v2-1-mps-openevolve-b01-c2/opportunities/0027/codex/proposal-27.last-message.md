MECHANISM: Ninth noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, 8, and 9 will produce a 1,601-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the verified eight-row gauge set with row 9, omitting its final input coefficient and reconstructing it as zero through the existing learned linear reparameterization.

EVIDENCE: The current eight-row design achieved 99.99% accuracy with 1,602 parameters; adding row 9 is the smallest incremental reduction and continues to avoid row 3, whose inclusion previously reduced accuracy to 97.77%.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8)
=======
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9)
>>>>>>> REPLACE