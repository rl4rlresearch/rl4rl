MECHANISM: Center-biased translation test-time ensembling

HYPOTHESIS: Increasing the unshifted crop weight from 3 to 4 will exceed 9,344 correct predictions by reducing boundary errors introduced by replicate-padded shifted views while retaining their ensemble benefit.

INTENDED_EDIT: Change only the five-crop TTA weights so the clean centered view receives half of the total crop weight instead of three-sevenths.

EVIDENCE: The current position-sensitive model achieves the best observed result at 9,344 correct, while removing spatial organization performed substantially worse; this motivates a conservative inference-only refinement that gives the spatially aligned view slightly more influence without disturbing the verified training and 0.98 EMA procedure.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (4.0, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE