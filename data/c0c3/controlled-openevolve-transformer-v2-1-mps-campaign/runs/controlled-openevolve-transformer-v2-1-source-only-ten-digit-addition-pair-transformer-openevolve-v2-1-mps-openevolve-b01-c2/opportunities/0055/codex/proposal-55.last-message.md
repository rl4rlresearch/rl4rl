MECHANISM: Balanced cross-head value-row normalized-input gauge

HYPOTHESIS: Adding value row 20 to the verified 1,580-parameter gauges will produce a 1,579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then omit the final normalized-input coefficient from value row 20, the first value coordinate of the second head.

EVIDENCE: Value rows 16 and 17 reached 99.94% and 99.83%, while extending within the first head to row 18 collapsed to 22.58%; the successful cross-head balancing of query rows 4 and 5 motivates testing the corresponding value coordinate in the second head.

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
        # Retain the verified balanced query, first-key, and first-head value
        # gauges, then test the corresponding value coordinate in head two.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
            2 * d_model + head_dim,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias while gauge-fixing two normalized-input QKV rows.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Key and value biases remain gauge-fixed while selected normalized-
        # input coefficients are balanced across attention heads.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE