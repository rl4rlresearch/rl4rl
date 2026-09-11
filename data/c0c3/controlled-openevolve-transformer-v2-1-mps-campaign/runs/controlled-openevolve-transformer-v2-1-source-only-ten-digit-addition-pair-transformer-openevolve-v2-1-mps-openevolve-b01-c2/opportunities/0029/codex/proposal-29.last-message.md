MECHANISM: Eleventh noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` row 11 in addition to rows 0, 1, 2, and 4–10 will produce a 1,599-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the verified noncontiguous gauge set through rows 7–11, omitting each selected row’s final input coefficient while continuing to leave sensitive row 3 ungauged.

EVIDENCE: The 1,600-parameter design gauging rows 0, 1, 2, and 4–10 achieved 99.91% accuracy; adding row 11 is the smallest untested reduction that preserves the empirically important exclusion of row 3.

<<<<<<< SEARCH
        self.gauged_rows = (0, 1, 2, 4, 5, 6)
=======
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11)
>>>>>>> REPLACE