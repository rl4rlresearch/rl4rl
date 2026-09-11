MECHANISM: Single-rounding order-preserving logit calibration

HYPOTHESIS: Retrying the unresolved single-rounding implementation at the proven 76.512% temperature will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Max-center accumulated ensemble logits before applying normalization and temperature sharpening in one multiplication.

EVIDENCE: The 76.512% design produced the best observed cross-entropy; all finer temperature probes were worse, while the only single-rounding test timed out and remains unresolved.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (84.0 / 76.511644)
=======
        centered_logits = logit_sum - logit_sum.amax(dim=1, keepdim=True)
        return centered_logits * (84.0 / (13.2124453125 * 76.512))
>>>>>>> REPLACE