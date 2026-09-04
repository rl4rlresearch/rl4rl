MECHANISM: Compressed linear flip-loss curriculum

HYPOTHESIS: Linearly increasing ensemble supervision from 0.60 to 0.90 will exceed 9,330 correct predictions by preserving the beneficial temporal shift while avoiding the endpoint extremes implicated by the weaker cosine curriculum.

INTENDED_EDIT: Replace the cosine 0.50-to-1.00 schedule with a same-average linear 0.60-to-0.90 schedule; retain the proven architecture, optimizer, and static 0.02 label smoothing.

EVIDENCE: The linear 0.50-to-1.00 curriculum achieved 9,330 correct, versus 9,325 for static 0.75 and 9,322 for the more endpoint-concentrated cosine schedule. A compressed linear schedule isolates whether the temporal progression is beneficial without excessive early view regularization or complete late removal of view supervision.

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.60 + 0.30 * progress
>>>>>>> REPLACE