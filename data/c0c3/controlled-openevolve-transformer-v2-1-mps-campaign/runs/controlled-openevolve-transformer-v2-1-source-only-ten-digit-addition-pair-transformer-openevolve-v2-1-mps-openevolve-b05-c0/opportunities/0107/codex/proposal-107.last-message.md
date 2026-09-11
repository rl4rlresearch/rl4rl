MECHANISM: Eighth LayerNorm-null key-projection quotient

HYPOTHESIS: Gauge-fixing the remaining key-projection row will reduce the verified model from 1604 to 1603 parameters while retaining at least 99% accuracy, because all seven previously quotient-fixed key rows train successfully and the eighth has the same softmax-invisible LayerNorm-null symmetry.

INTENDED_EDIT: Add key row `d_model + 3` to the existing gauge-fixed QKV rows so the current virtual AdamW and gradient-clipping machinery reconstructs its omitted coordinate.

EVIDENCE: The verified 1604-parameter design reached 99.92% accuracy while already removing seven key-projection coordinates through this exact mechanism; extending that established gauge family avoids the attention-output and LayerNorm changes that repeatedly timed out.

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
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
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
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
>>>>>>> REPLACE