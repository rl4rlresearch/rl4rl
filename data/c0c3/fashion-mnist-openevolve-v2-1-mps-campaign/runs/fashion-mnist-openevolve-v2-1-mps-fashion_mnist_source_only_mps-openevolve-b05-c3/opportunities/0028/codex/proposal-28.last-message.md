MECHANISM: Validated 39/64 augmentation curriculum

HYPOTHESIS: Starting center/cardinal augmentation after 39/64 of training will improve on the current 9,141 correct predictions and reproduce the best observed 9,167-result region.

INTENDED_EDIT: Replace the current three-quarter broad-translation phase with the best verified 39/64 transition, preserving all other settings.

EVIDENCE: The otherwise identical 39/64 design achieved 9,167 correct and 0.23885 cross-entropy, outperforming the current three-quarter design’s 9,141 correct and 0.24300 cross-entropy.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 64 < total_steps * 39:
>>>>>>> REPLACE