MECHANISM: Seventh noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, and 7 will produce a 1,603-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the successful six-row gauge set with row 7, omitting its final input coefficient and reconstructing it as zero through the existing reparameterization.

EVIDENCE: Gauging rows 0, 1, 2, 4, 5, and 6 achieved 99.94% accuracy with 1,604 parameters; adding row 7 is the smallest informative reduction while continuing to exclude sensitive row 3.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4)
=======
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7)
>>>>>>> REPLACE