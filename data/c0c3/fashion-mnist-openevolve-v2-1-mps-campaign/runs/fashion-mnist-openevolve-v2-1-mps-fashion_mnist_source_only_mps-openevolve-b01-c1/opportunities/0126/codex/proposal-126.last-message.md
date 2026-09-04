MECHANISM: Cosine-decayed label smoothing with a 0.009 floor

HYPOTHESIS: Ending smoothing at 0.009 will exceed 9,259 correct predictions by retaining slightly more late regularization than the regressed 0.007 floor while remaining closer to the best 0.008 result than the 0.010 floor.

INTENDED_EDIT: Raise the cosine schedule’s final smoothing from 0.008 to 0.009 while preserving its initial value of 0.020.

EVIDENCE: The 0.008 floor achieved 9,259 correct, while 0.007 fell to 9,256 and 0.010 reached 9,258; 0.009 is the nearest untested value on the better-performing, more-regularized side of the current optimum.

<<<<<<< SEARCH
    smoothing = 0.008 + 0.006 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.009 + 0.0055 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE