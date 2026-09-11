MECHANISM: Symmetric upper-side second-moment interpolation

HYPOTHESIS: AdamW beta2=0.9625 will exceed 9,283 correct predictions by refining the unexplored upper side of the narrow optimum around beta2=0.96.

INTENDED_EDIT: Increase AdamW beta2 from 0.97 to 0.9625 while preserving all other model, training, augmentation, schedule, and evaluation settings.

EVIDENCE: Beta2=0.96 achieved the best result at 9,283 correct, while the equally spaced lower-side value 0.9575 achieved 9,280 and beta2=0.97 achieved 9,278; testing 0.9625 completes the closest symmetric bracket around the observed optimum.

<<<<<<< SEARCH
        betas=(0.9, 0.97),
=======
        betas=(0.9, 0.9625),
>>>>>>> REPLACE