MECHANISM: Sixth virtual-AdamW key-projection LayerNorm-null quotient

HYPOTHESIS: Gauge-fixing key row `d_model + 1` will reduce the verified model from 1609 to 1608 parameters while retaining at least 99% accuracy, because it uses the same exact LayerNorm-null key-weight gauge and virtual optimizer already applied successfully to five other key rows.

INTENDED_EDIT: Add the second row of the first attention head to `GaugeFixedQKV.fixed_rows`; the existing generalized reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically cover it.

EVIDENCE: The verified 1609-parameter model achieved 99.97% accuracy while quotienting five key-projection rows with this machinery, making another independent remaining key row the most direct evidence-backed reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with five softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
=======
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
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
>>>>>>> REPLACE