MECHANISM: Conservative extension of inference-aligned augmentation

HYPOTHESIS: Beginning cardinal augmentation after 77/128 of training will exceed 9,167 correct predictions by extending the beneficial inference-aligned phase beyond the best completed 25/64 phase without jumping fully to the timed-out 13/32 phase.

INTENDED_EDIT: Use broad ±2 translations for the first 77/128 of training and center/cardinal one-pixel translations thereafter.

EVIDENCE: Extending the cardinal phase from 1/4 through 25/64 monotonically improved correct predictions from 9,141 to 9,167; 77/128 is the midpoint between the best completed 25/64 duration and the timed-out 13/32 attempt.

<<<<<<< SEARCH
    if step * 8 < total_steps * 5:
=======
    if step * 128 < total_steps * 77:
>>>>>>> REPLACE