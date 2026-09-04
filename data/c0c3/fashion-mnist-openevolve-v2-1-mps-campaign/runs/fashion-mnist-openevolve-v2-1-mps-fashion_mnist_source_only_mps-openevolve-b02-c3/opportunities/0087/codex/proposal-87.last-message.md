MECHANISM: Vertical-favoring translation ensemble

HYPOTHESIS: Favoring vertical shifts will exceed 9,344 correct predictions because the equal-and-opposite horizontal-favoring perturbation reduced correctness to 9,339.

INTENDED_EDIT: Restore the proven center weight of 3.0 and, while preserving total ensemble weight, assign each vertical crop weight 1.25 and each horizontal crop weight 0.75.

EVIDENCE: Symmetric weight-3 TTA achieved 9,344 correct, whereas reallocating 0.25 from each vertical crop to each horizontal crop achieved only 9,339; testing the opposite direction is the most informative follow-up.

<<<<<<< SEARCH
        crop_weights = (2.5, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 1.25, 1.25, 0.75, 0.75)
>>>>>>> REPLACE