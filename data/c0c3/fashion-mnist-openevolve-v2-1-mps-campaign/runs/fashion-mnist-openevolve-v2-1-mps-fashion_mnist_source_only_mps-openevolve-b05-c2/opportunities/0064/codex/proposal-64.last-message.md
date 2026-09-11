MECHANISM: Proven rational temperature restoration

HYPOTHESIS: Restoring the exact Reference Design 3 calibration will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237594985961914 to approximately 0.22237433319091796.

INTENDED_EDIT: Replace the current 76.74% effective denominator with the best verified 76.512% rational scaling, without changing class ordering.

EVIDENCE: Reference Design 3 achieved the best observed result—9,192 correct with 0.22237433319091796 cross-entropy—and both immediately adjacent temperature probes were worse.

<<<<<<< SEARCH
        return normalized_logits * (1400.0 / 1279.0)
=======
        return normalized_logits * (10500.0 / 9564.0)
>>>>>>> REPLACE