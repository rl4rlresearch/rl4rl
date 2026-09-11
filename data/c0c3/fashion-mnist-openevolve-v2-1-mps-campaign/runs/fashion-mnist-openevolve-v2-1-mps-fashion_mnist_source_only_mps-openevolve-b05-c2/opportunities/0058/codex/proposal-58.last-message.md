MECHANISM: Single-rounding order-preserving logit calibration

HYPOTHESIS: Using the proven 76.512% temperature while collapsing normalization and sharpening into one scaling operation will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Max-center the accumulated logits before applying one combined normalization-temperature scale, eliminating an intermediate tensor division and its rounding error.

EVIDENCE: The 76.512% design has the best observed cross-entropy; three finer denominator probes were worse, suggesting arithmetic precision is now more informative than further temperature interpolation.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (84.0 / 76.512356)
=======
        centered_logits = logit_sum - logit_sum.amax(dim=1, keepdim=True)
        return centered_logits * (
            84.0 / (13.2124453125 * 76.512)
        )
>>>>>>> REPLACE