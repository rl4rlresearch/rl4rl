MECHANISM: Moderately center-weighted test-time augmentation

HYPOTHESIS: Weighting the unshifted validation view twice as heavily as each cardinal translation will exceed 9,290 correct predictions by retaining translation robustness while emphasizing the original image geometry.

INTENDED_EDIT: Duplicate the centered view in the evaluation ensemble, changing its weight from 20% to 33.3% while leaving training and all other settings unchanged.

EVIDENCE: Cardinal translation-flip averaging previously improved validation correct from 9,138 to 9,167, while adding diagonal views reduced it; this shows evaluation-view composition affects accuracy and motivates a conservative weighting refinement.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
=======
        views = (
            images,
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
>>>>>>> REPLACE