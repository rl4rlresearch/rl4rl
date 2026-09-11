MECHANISM: Pre-scale mean-centered logit calibration

HYPOTHESIS: Mean-centering before applying the best-verified float32 coefficient will preserve 9,192 correct predictions and reduce validation cross-entropy below 0.22237433319091796 by reducing multiplication-rounding error.

INTENDED_EDIT: Restore the best verified sharpening coefficient and replace max-centering with order-preserving mean-centering.

EVIDENCE: Both adjacent coefficients were worse, while post-scale max-centering and log-softmax canonicalization also failed to improve; mean-centering is an untested equivalent calibration path that reduces centered-logit magnitude.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * (84.0 / 76.511644)
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.mean(
            dim=1, keepdim=True
        )
        return normalized_logits * 1.0978671312332153
>>>>>>> REPLACE