MECHANISM: Order-locked midpoint temperature sharpening

HYPOTHESIS: Anchoring predictions to the proven 84%-denominator logits and sharpening them to 76% will retain 9,192 correct predictions while reducing cross-entropy below 0.22403448944091797.

INTENDED_EDIT: Retry the timed-out 76% calibration probe, max-centering the proven 84%-denominator logits before positive scaling to preserve their class ordering.

EVIDENCE: The 84% denominator achieved 9,192 correct with 0.22403448944091797 cross-entropy, while 68% lost one prediction; the previously attempted order-locked 76% midpoint timed out and therefore remains the most informative unresolved probe.

<<<<<<< SEARCH
        return logit_sum / 14.4707734375
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (21.0 / 19.0)
>>>>>>> REPLACE