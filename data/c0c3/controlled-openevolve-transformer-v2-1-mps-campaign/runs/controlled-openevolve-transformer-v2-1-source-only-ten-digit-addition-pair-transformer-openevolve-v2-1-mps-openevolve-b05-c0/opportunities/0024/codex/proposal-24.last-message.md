MECHANISM: Fourth quotient-aware key gauge within the empirically robust second attention head

HYPOTHESIS: Removing the softmax-invisible coordinate from the third key row of the second head will reduce the model from 1630 to 1629 parameters while retaining at least 99% accuracy, because two removals in that head succeeded whereas the second removal in the first head failed.

INTENDED_EDIT: Add the third key row of the second attention head to `GaugeFixedQKV.fixed_rows`; the existing generalized virtual AdamW and gradient-clipping logic will handle the fourth omitted coordinate.

EVIDENCE: The 1630-parameter model reached 99.91% after removing two coordinates from the second head, while placing the second within-head removal in the first head reached only 87.60%; this motivates continuing the smallest one-coordinate titration in the empirically successful head.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with three softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
        )
=======
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
>>>>>>> REPLACE