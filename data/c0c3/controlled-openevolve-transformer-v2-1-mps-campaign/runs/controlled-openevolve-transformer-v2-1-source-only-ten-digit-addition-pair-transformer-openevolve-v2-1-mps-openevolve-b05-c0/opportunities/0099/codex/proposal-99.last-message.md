MECHANISM: Fourth attention-output common-shift quotient

HYPOTHESIS: Gauge-fixing attention-output column 3 will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because columns 7, 6, and 4 already support independent residual-shift symmetries and column 3 avoids the feature-5 virtual value-bias pathway implicated in the failed column-5 trial.

INTENDED_EDIT: Add a fourth attention-output projection gauge at row 4, column 3 using the existing generic reconstruction, virtual AdamW, and gradient-clipping machinery.

EVIDENCE: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy with 1604 parameters, whereas using column 5 as the third gauge achieved only 72.98%; testing the next non-feature-5 column is the closest controlled extension of the successful design.

<<<<<<< SEARCH
class GaugeFixedAttentionOutput(nn.Module):
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
class GaugeFixedAttentionOutput(nn.Module):
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
            in_features - 5,
        )
>>>>>>> REPLACE