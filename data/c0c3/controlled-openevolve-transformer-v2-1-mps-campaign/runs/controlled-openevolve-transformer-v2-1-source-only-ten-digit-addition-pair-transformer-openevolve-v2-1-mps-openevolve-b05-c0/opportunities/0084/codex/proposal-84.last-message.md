MECHANISM: Seventh virtual-AdamW key-projection LayerNorm-null quotient

HYPOTHESIS: Gauge-fixing key row `d_model + 2` will reduce the verified model from 1608 to 1607 parameters while retaining at least 99% accuracy, because the identical virtual-optimizer gauge already succeeds across six key rows.

INTENDED_EDIT: Add the third row of the first attention head to `GaugeFixedQKV.fixed_rows`; existing reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically generalize to it.

EVIDENCE: Adding key row `d_model + 1` reduced the model from 1609 to 1608 parameters with 99.95% accuracy, directly validating another independent application of this exact gauge.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with six softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + second_offset,
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with seven softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + second_offset,
>>>>>>> REPLACE