MECHANISM: Nonadjacent first-head LayerNorm key gauge

HYPOTHESIS: Removing the softmax-invisible coordinate from the third key row of the first attention head will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, testing whether the earlier first-head failure was specific to its second key row rather than to additional gauges in that head.

INTENDED_EDIT: Add the third key row of the first head to `GaugeFixedQKV.fixed_rows`; the generalized virtual-coordinate AdamW and gradient-clipping logic will handle the sixth omitted coordinate.

EVIDENCE: All four key rows of the second head were removed successfully, reaching 99.40% at 1628 parameters, while adding the adjacent second row of the first head previously produced 87.60%; the untested nonadjacent third row is the smallest informative way to distinguish row-specific sensitivity from a general first-head limit.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """QKV projection with five softmax-invisible coordinates removed."""
=======
class GaugeFixedQKV(nn.Module):
    """QKV projection with six softmax-invisible coordinates removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fixed_rows = (
            d_model,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
=======
        self.fixed_rows = (
            d_model,
            d_model + 2,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
>>>>>>> REPLACE