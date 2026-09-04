MECHANISM: Fifth quotient-aware LayerNorm key gauge in the robust second attention head

HYPOTHESIS: Removing the remaining softmax-invisible coordinate from the fourth key row of the second head will reduce the model from 1629 to 1628 parameters while retaining at least 99% accuracy, because three successive removals in that head met the threshold and the existing virtual-coordinate AdamW preserves each omitted coordinate’s full-space dynamics.

INTENDED_EDIT: Add the fourth key row of the second attention head to `GaugeFixedQKV.fixed_rows`; the generalized optimizer and gradient-clipping logic already supports the fifth omitted coordinate.

EVIDENCE: The 1629-parameter model retained 99.11% after three key-coordinate removals in the second head, while the second removal in the first head failed at 87.60%; continuing the empirically robust head by one coordinate is the smallest informative reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with four softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
        )
=======
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
>>>>>>> REPLACE