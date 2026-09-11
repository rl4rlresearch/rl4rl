MECHANISM: Half-step refinement of the label-smoothing endpoint

HYPOTHESIS: Ending smoothing at 0.0085 will exceed 9,262 correct predictions by preserving the stronger classification behavior of the 0.009 endpoint while moving toward the lower cross-entropy observed at 0.008.

INTENDED_EDIT: Change the cosine label-smoothing schedule from 0.020→0.009 to 0.020→0.0085, leaving all other training behavior unchanged.

EVIDENCE: The 0.009 endpoint achieved the best result with 9,262 correct, while 0.008 was the better adjacent endpoint than 0.010 in both correct predictions and cross-entropy; 0.0085 is the most informative untested refinement on that side.

<<<<<<< SEARCH
    smoothing = 0.009 + 0.0055 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.0085 + 0.00575 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE