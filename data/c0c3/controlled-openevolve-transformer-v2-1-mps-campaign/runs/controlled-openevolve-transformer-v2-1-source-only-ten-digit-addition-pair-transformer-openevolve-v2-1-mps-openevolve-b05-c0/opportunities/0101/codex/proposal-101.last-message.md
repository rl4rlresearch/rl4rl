MECHANISM: Alternate-column fourth attention-output common-shift quotient

HYPOTHESIS: Gauge-fixing attention-output column 2 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because columns 7, 6, and 4 already support independent residual-shift quotients and column 2 avoids both the feature-5 value-bias pathway and the repeatedly timed-out column-3 configuration.

INTENDED_EDIT: Add a fourth attention-output projection gauge at row 4, column 2 using the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery.

EVIDENCE: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy with 1604 parameters; column 5 failed at 72.98%, while column 3 produced two timeouts rather than an accuracy result, motivating the next unused column that does not overlap feature 5.

<<<<<<< SEARCH
    """Attention output projection with three residual-shift gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        fixed_row = min(4, out_features - 1)
        self.fixed_weight_rows = (fixed_row, fixed_row, fixed_row)
        self.fixed_weight_columns = (
            in_features - 1,
            in_features - 2,
            in_features - 4,
        )
=======
    """Attention output projection with four residual-shift gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        fixed_row = min(4, out_features - 1)
        self.fixed_weight_rows = (
            fixed_row,
            fixed_row,
            fixed_row,
            fixed_row,
        )
        self.fixed_weight_columns = (
            in_features - 1,
            in_features - 2,
            in_features - 4,
            in_features - 6,
        )
>>>>>>> REPLACE