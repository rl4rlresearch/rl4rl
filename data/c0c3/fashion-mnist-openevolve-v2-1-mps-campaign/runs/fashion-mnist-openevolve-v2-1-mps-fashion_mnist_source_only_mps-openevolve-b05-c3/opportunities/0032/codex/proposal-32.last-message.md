MECHANISM: Validated 39/64 augmentation curriculum

HYPOTHESIS: Starting center/cardinal augmentation after 39/64 of training will exceed the current 9,141 correct predictions and reproduce the reference result of 9,167 correct.

INTENDED_EDIT: Replace the three-quarter broad-translation phase with the best verified 39/64 transition while preserving all other settings.

EVIDENCE: Reference Design 1 achieved 9,167 correct and 0.23885 cross-entropy, outperforming the otherwise identical three-quarter design’s 9,141 correct and 0.24300 cross-entropy.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE