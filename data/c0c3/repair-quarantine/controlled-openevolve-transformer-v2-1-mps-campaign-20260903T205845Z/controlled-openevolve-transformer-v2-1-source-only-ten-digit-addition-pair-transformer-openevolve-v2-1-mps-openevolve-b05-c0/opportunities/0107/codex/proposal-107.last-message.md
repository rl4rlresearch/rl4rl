MECHANISM: Complete LayerNorm-null key-row gauge fixing

HYPOTHESIS: Fixing the remaining key row `d_model + 3` will reduce the model from 1607 to 1606 learned parameters while preserving at least 99% accuracy, because it has the same softmax-invisible LayerNorm-null direction as the seven already-fixed key rows.

INTENDED_EDIT: Remove one additional QKV parameter by extending the existing virtual-optimizer gauge to the sole unfixed key row.

EVIDENCE: The current seven-row gauge-fix design verified at 0.9997 accuracy with 1607 parameters, supporting another symmetry-equivalent key-row removal.

<<<<<<< SEARCH
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
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with eight softmax-invisible coordinates removed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        second_offset = d_model // n_head if n_head > 1 else 1
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + second_offset,
>>>>>>> REPLACE