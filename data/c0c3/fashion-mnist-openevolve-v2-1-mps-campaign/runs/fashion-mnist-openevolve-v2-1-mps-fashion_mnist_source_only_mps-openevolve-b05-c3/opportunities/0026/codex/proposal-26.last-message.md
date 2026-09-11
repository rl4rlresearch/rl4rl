MECHANISM: Minimal extension of inference-aligned augmentation

HYPOTHESIS: Beginning cardinal augmentation after 311/512 of training will exceed 9,167 correct predictions by extending the consistently beneficial inference-aligned phase three optimizer steps beyond the best completed 39/64 configuration.

INTENDED_EDIT: Use broad ±2 translations for the first 311/512 of training and center/cardinal one-pixel translations thereafter, retaining all other validated settings.

EVIDENCE: Extending the cardinal phase from one-quarter through 25/64 monotonically improved validation correct from 9,141 to 9,167; the attempted 155/256 boundary timed out without negative accuracy evidence, so 311/512 is the smallest intermediate extension toward it.

<<<<<<< SEARCH
    if step * 8 < total_steps * 5:
=======
    if step * 512 < total_steps * 311:
>>>>>>> REPLACE