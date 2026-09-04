MECHANISM: Directed adjacent-float reciprocal calibration

HYPOTHESIS: Increasing the reciprocal-division temperature by one float32 ULP will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796 through a previously untested division-rounding path.

INTENDED_EDIT: Replace rational multiplication with division by the next representable float32 temperature above the reciprocal of the best verified sharpening coefficient.

EVIDENCE: Reciprocal division preserved 9,192 correct predictions, while the lower-sharpening multiplication probe degraded less than the higher-sharpening probe; this motivates moving the division temperature one ULP toward lower sharpening.

<<<<<<< SEARCH
        return normalized_logits * (10500.0 / 9564.0)
=======
        temperature = normalized_logits.new_tensor(
            1.0 / 1.0978671312332153
        )
        temperature = torch.nextafter(
            temperature, temperature.new_tensor(float("inf"))
        )
        return normalized_logits / temperature
>>>>>>> REPLACE