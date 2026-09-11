MECHANISM: Extended inference-aligned augmentation curriculum

HYPOTHESIS: Moving the broad-to-cardinal transition from 39/64 to 19/32 of training will exceed 9,167 correct predictions by continuing the observed improvement from progressively longer terminal phases.

INTENDED_EDIT: Use broad ±2 translations for the first 19/32 of training, then center/cardinal one-pixel translations for the remaining 13/32.

EVIDENCE: Extending the inference-aligned phase improved correct predictions monotonically from 9,141 at one quarter to 9,155 at one third, 9,163 at three eighths, and 9,167 at 25/64; the next equal 1/64 extension is the most focused continuation.

<<<<<<< SEARCH
    if step * 4 < total_steps * 3:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE