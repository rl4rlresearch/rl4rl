MECHANISM: Eighth virtual-AdamW key-projection LayerNorm-null quotient

HYPOTHESIS: Gauge-fixing key row `d_model + 3` will reduce the verified model from 1607 to 1606 parameters while retaining at least 99% accuracy, because the identical virtual-optimizer gauge has succeeded across seven key rows, including the three adjacent rows of the first attention head.

INTENDED_EDIT: Add the remaining row of the first attention head to `GaugeFixedQKV.fixed_rows`; the existing reconstruction, virtual-gradient recovery, AdamW moments, and gauge-aware clipping automatically cover it.

EVIDENCE: Adding key row `d_model + 2` reduced the model from 1608 to 1607 parameters with 99.97% accuracy, directly validating the same gauge on the adjacent first-head key row.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with seven softmax-invisible coordinates removed."""
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with eight softmax-invisible coordinates removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + second_offset,
=======
        self.fixed_rows = (
            d_model,
            d_model + 1,
            d_model + 2,
            d_model + 3,
            d_model + second_offset,
>>>>>>> REPLACE