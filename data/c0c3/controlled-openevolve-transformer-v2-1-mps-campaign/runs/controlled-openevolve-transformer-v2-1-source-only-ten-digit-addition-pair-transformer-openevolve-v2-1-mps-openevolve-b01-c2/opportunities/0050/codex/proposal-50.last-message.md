MECHANISM: Balanced cross-head key-row LayerNorm-nullspace gauge

HYPOTHESIS: Extending the verified 1,582-parameter design with the corresponding first key row of the second attention head will produce a 1,581-parameter model with at least 99% accuracy.

INTENDED_EDIT: Gauge query rows 0, 1, 4, and 5 and key rows 8 and 12, balancing both query and key gauges across the two attention heads.

EVIDENCE: Reference Design 3 achieved 99.93% accuracy at 1,582 parameters after adding key row 8 to four balanced query gauges; adding key row 12 is the smallest cross-head-balanced extension and avoids the failed strategy of constraining another query coordinate.

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1)
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        # Balance two query-row gauges and one key-row gauge across each head.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + head_dim,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias while gauge-fixing two normalized-input QKV rows.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Retain query bias while gauge-fixing balanced query and key rows.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves both selected row functions.
=======
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves every selected row function.
>>>>>>> REPLACE