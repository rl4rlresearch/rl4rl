MECHANISM: Mean-preserving linear label-smoothing anneal

HYPOTHESIS: Replacing the best verified cosine 0.04→0 smoothing anneal with a full-horizon linear anneal of identical endpoints and mean will exceed 9,240 correct predictions by retaining more regularization during later updates without weakening the empirically optimal initial smoothing.

INTENDED_EDIT: Adopt the verified 1.296352 calibration and replace the current 0.05→0.01 cosine smoothing with a linear 0.04→0 schedule; architecture, optimizer, learning-rate schedule, and ensemble weighting remain unchanged.

EVIDENCE: Cosine 0.04→0 achieved the best verified score at 9,240 correct, while ending smoothing halfway fell to 9,233 and changing the initial amplitude to 0.03 or 0.05 also lost correctness; a full-horizon linear schedule isolates decay curvature while preserving the successful endpoints and average smoothing.

<<<<<<< SEARCH
        return 1.295 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.04 * (1.0 - progress)
>>>>>>> REPLACE