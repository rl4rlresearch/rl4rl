MECHANISM: Sixth LayerNorm-null key-projection gauge

HYPOTHESIS: Fixing key-projection coordinate 3 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because this softmax-invisible direction uses the same virtual AdamW quotient already supporting five removed key coordinates.

INTENDED_EDIT: Add key row 3 to the gauge-fixed QKV rows; the existing reconstruction, gradient clipping, and optimizer logic automatically handles the additional omitted coordinate.

EVIDENCE: The five-coordinate key quotient is present in the 99.93% design. Among the remaining key coordinates, coordinate 3 has the strongest evidence: its token-position gauge reached 95.46% and its LayerNorm-scale gauge reached 98.67%, while the tested key coordinate 1 reached only 71.24%.

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
            d_model + 3,
            d_model + second_offset,
>>>>>>> REPLACE