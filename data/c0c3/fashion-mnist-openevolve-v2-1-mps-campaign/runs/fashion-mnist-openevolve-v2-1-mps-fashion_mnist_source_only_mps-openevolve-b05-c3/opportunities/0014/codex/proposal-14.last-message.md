MECHANISM: Intermediate-duration inference-aligned augmentation

HYPOTHESIS: Beginning center/cardinal one-pixel augmentation after 60% of training will exceed 9,155 correct predictions by extending the beneficial inference-matched phase beyond one-third while retaining more broad ±2 translation exposure than the unverified half-run configuration.

INTENDED_EDIT: Run broad ±2 translation augmentation for the first three-fifths of training and center/cardinal augmentation for the final two-fifths.

EVIDENCE: Extending the cardinal phase from the final quarter to the final third improved validation correct from 9,141 to 9,155 and lowered cross-entropy; the attempted final-half phase timed out, making a two-fifths phase the closest informative intermediate test.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 5 < total_steps * 3:
>>>>>>> REPLACE