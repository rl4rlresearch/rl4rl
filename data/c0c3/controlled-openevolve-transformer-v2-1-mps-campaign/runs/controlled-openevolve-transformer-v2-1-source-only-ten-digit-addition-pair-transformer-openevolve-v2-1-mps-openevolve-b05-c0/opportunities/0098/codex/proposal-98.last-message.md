MECHANISM: Alternate-column third attention-output common-shift quotient

HYPOTHESIS: Gauge-fixing attention-output column 4 instead of the previously failed column 5 will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because columns 7 and 6 already support this exact symmetry and column 4 avoids overlap with the virtual value-bias pathway on feature 5.

INTENDED_EDIT: Add a third attention-output projection gauge at row 4, column 4 while leaving the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery unchanged.

EVIDENCE: Two attention-output quotients reached 99.99% accuracy, but extending them to the final three columns failed at 72.98%; the failed third coordinate was column 5, which the current design also uses for virtual value-bias compensation, motivating an alternate independent column.

<<<<<<< SEARCH
    """Attention output projection with two residual-shift gauges fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        fixed_row = min(4, out_features - 1)
        self.fixed_weight_rows = (fixed_row, fixed_row)
        self.fixed_weight_columns = (in_features - 1, in_features - 2)
=======
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
>>>>>>> REPLACE