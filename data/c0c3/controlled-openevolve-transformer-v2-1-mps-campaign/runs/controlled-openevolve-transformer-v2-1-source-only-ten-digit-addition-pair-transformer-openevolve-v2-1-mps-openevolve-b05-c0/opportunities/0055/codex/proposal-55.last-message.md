MECHANISM: Additional LayerNorm-null key-projection gauge

HYPOTHESIS: Fixing key-projection coordinate 1 will reduce the model from 1617 to 1616 parameters while retaining at least 99% accuracy, because its omitted weight is an exact softmax-invisible quotient and the existing virtual AdamW path preserves its optimization geometry.

INTENDED_EDIT: Add key row 1 to the five already gauge-fixed QKV rows; the existing dynamic virtual-gradient, clipping, and optimizer logic automatically handles the sixth coordinate.

EVIDENCE: The current 1617-parameter design achieves 99.83% while simultaneously fixing five coordinates through this same key-projection quotient. Coordinate 1 is the strongest remaining candidate because its analogous second-LayerNorm scale gauge retained 99.81% accuracy.

<<<<<<< SEARCH
    """QKV projection with five softmax-invisible coordinates removed."""
=======
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
            d_model + 1,
            d_model + second_offset,
            d_model + second_offset + 1,
            d_model + second_offset + 2,
            d_model + second_offset + 3,
        )
>>>>>>> REPLACE