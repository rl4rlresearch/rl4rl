MECHANISM: Shift-favoring translation test-time ensembling

HYPOTHESIS: Reducing the centered-crop weight from 3 to 2 will exceed 9,344 correct predictions by giving more influence to shifted views, reversing the direction that degraded accuracy.

INTENDED_EDIT: Change only the five-crop TTA weights so the centered view receives one-third of total crop weight instead of three-sevenths.

EVIDENCE: Increasing the center weight from 3 to 4 reduced correctness from 9,344 to 9,338, providing direct evidence that emphasizing the aligned view is harmful and motivating the opposite localized adjustment.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.0, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE