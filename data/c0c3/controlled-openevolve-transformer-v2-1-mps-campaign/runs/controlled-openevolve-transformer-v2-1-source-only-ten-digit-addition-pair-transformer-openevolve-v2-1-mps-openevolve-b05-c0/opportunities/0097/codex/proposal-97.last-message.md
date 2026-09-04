MECHANISM: Third attention-output common-shift quotient

HYPOTHESIS: Removing a third common-output coordinate from the attention projection will reduce the verified model from 1605 to 1604 parameters while retaining at least 99% accuracy, because each projection input column has an independent residual-stream shift symmetry and the first two attention-output quotients achieved 99.98% and 99.99% accuracy.

INTENDED_EDIT: Gauge-fix row 4 in the final three attention-output projection columns; the existing generic reconstruction, virtual AdamW, and gradient-clipping logic already supports multiple omitted coordinates.

EVIDENCE: Extending the attention-output quotient from one to two coordinates reduced parameters from 1606 to 1605 and improved verified accuracy from 99.98% to 99.99%, directly supporting another independent column in the same projection.

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
            in_features - 3,
        )
>>>>>>> REPLACE