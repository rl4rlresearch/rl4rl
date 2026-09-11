MECHANISM: Cosine-decayed label smoothing with a locally optimized 0.007 floor

HYPOTHESIS: Ending smoothing at 0.007 will exceed 9,259 correct predictions by slightly sharpening late decision boundaries while remaining near the cross-entropy optimum implied by the 0.008, 0.010, and 0.013 results.

INTENDED_EDIT: Lower only the cosine smoothing schedule’s final value from 0.008 to 0.007, preserving the initial value of 0.02 and all other training behavior.

EVIDENCE: The 0.008 floor achieved the best result at 9,259 correct and lower cross-entropy than 0.010 and 0.013, while a zero floor regressed sharply; the observed cross-entropy trend places the most informative next probe just below 0.008 rather than near zero.

<<<<<<< SEARCH
    smoothing = 0.008 + 0.006 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.007 + 0.0065 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE