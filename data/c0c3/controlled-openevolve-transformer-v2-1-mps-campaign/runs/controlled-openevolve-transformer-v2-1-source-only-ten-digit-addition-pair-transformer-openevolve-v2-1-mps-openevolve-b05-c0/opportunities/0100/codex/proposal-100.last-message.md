MECHANISM: Fourth attention-output common-shift quotient with final-only validation

HYPOTHESIS: Gauge-fixing attention-output column 3 will reduce the model from 1604 to 1603 parameters while retaining at least 99% accuracy; evaluating only after the final update will avoid the timeout seen in the previous column-3 trial without changing its 5000-step optimization.

INTENDED_EDIT: Add a fourth residual-shift gauge at row 4, column 3 and remove intermediate validation passes while still producing both final-step best and last checkpoints.

EVIDENCE: Three attention-output gauges at columns 7, 6, and 4 achieved 99.92% accuracy. The direct column-3 extension timed out rather than failing accuracy, so completing that controlled extension with lower validation overhead is the most informative next test.

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
        self.fixed_weight_rows = (fixed_row, fixed_row, fixed_row, fixed_row)
        self.fixed_weight_columns = (
            in_features - 1,
            in_features - 2,
            in_features - 4,
            in_features - 5,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE