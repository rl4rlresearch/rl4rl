MECHANISM: Bracketed shift-augmentation duration refinement

HYPOTHESIS: Extending deterministic shift augmentation from 50% to 55% of training will exceed 9,256 correct predictions by balancing the accuracy losses observed at both 40% and approximately 69% exposure.

INTENDED_EDIT: End shift augmentation after 55% of optimizer steps while preserving all other training and inference behavior.

EVIDENCE: Reducing augmentation exposure to 40% produced 9,244 correct, while extending it to 20/29 produced 9,246 versus 9,256 at 50%; interpolation between these bracketed results motivates a small extension near 55%.

<<<<<<< SEARCH
    if step < total_steps // 2:
=======
    if step < (11 * total_steps) // 20:
>>>>>>> REPLACE