MECHANISM: Single-rounding fused ensemble calibration

HYPOTHESIS: Collapsing ensemble normalization and the best verified sharpening coefficient into one float32 multiplication will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Apply one combined scale directly to the accumulated ensemble logits, then max-center the calibrated result.

EVIDENCE: Reference Design 1 has the best observed cross-entropy using separate normalization and multiplication, while post-scale centering with separate operations was only slightly worse; eliminating the intermediate division rounding is the most informative untested equivalent path.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits / (1.0 / 1.0978671312332153)
=======
        calibrated_logits = logit_sum * (
            1.0978671312332153 / 13.2124453125
        )
        return calibrated_logits - calibrated_logits.amax(
            dim=1, keepdim=True
        )
>>>>>>> REPLACE