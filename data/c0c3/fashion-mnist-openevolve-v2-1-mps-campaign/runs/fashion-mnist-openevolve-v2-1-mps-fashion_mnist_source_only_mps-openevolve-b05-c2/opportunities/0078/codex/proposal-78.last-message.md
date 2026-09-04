MECHANISM: Pre-division raw-ensemble max-centering

HYPOTHESIS: Centering the accumulated logits before normalization will preserve all 9,192 predictions while reducing validation cross-entropy below 0.22237433319091796 by avoiding subtraction between separately rounded quotients.

INTENDED_EDIT: Move order-preserving max-centering from normalized logits to the raw ensemble sum, retaining the best verified float32 sharpening coefficient.

EVIDENCE: The current calibration has the best observed cross-entropy, while post-scale and mean-centering variants were worse; centering before division is an untested rounding path that preserves the same mathematical logits and class ordering.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * 1.0978671312332153
=======
        logit_sum = logit_sum - logit_sum.amax(dim=1, keepdim=True)
        normalized_logits = logit_sum / 13.2124453125
        return normalized_logits * 1.0978671312332153
>>>>>>> REPLACE