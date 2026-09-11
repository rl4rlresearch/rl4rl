MECHANISM: Center-biased crop-augmentation curriculum

HYPOTHESIS: Shortening shift augmentation from 50% to 40% of training will exceed 9,256 correct predictions by retaining useful translation robustness while increasing optimization on unshifted images.

INTENDED_EDIT: End deterministic shift augmentation after two-fifths of optimizer steps instead of one-half; preserve all other training and inference behavior.

EVIDENCE: Extending shift augmentation to 20/29 of training reduced validation_correct from 9,256 to 9,246, indicating that greater shifted-view exposure is harmful and motivating a conservative test in the opposite direction.

<<<<<<< SEARCH
    if step < total_steps // 2:
=======
    if 5 * step < 2 * total_steps:
>>>>>>> REPLACE