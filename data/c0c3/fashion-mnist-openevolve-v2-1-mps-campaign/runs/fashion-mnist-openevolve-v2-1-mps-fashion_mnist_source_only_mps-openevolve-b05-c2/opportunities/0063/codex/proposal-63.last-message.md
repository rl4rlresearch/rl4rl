MECHANISM: Proven rational temperature restoration

HYPOTHESIS: Restoring Reference Design 3’s exact rational calibration will preserve 9,192 correct predictions and reduce validation cross-entropy from 0.22237433738708495 to approximately 0.22237433319091796.

INTENDED_EDIT: Replace the current lower-side temperature counterprobe with the exact rational scaling of the best verified design.

EVIDENCE: Reference Design 3 achieved the best observed validation cross-entropy, 0.22237433319091796, with the same 9,192 correct predictions; both adjacent denominator probes were worse.

<<<<<<< SEARCH
        return normalized_logits * (84.0 / 76.511644)
=======
        return normalized_logits * (10500.0 / 9564.0)
>>>>>>> REPLACE