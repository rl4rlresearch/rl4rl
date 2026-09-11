MECHANISM: Remaining first-head LayerNorm key gauge

HYPOTHESIS: Removing the softmax-invisible coordinate from the fourth key row of the first attention head will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, testing whether optimization sensitivity is confined to the previously failed second and third rows.

INTENDED_EDIT: Add the fourth key row of the first head to `GaugeFixedQKV.fixed_rows`; the generalized virtual-coordinate AdamW and gradient-clipping logic will handle the sixth omitted coordinate.

EVIDENCE: Removing all four key coordinates from the second head succeeded at 99.40%, while additions from the second and third rows of the first head failed; its fourth row is the only untested exact key gauge and therefore the smallest informative next reduction.

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
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with six softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + second_offset - 1,
            d_model + second_offset,
>>>>>>> REPLACE