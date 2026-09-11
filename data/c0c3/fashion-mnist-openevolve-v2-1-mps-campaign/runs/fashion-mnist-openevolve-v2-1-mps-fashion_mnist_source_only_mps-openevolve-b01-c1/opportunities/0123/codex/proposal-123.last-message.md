MECHANISM: Cosine-decayed label smoothing with an interpolated 0.013 floor

HYPOTHESIS: Ending label smoothing at 0.013 will exceed 9,258 correct predictions by balancing the extra boundary sharpening of the 0.01 floor against the stronger late regularization of fixed 0.02 smoothing.

INTENDED_EDIT: Preserve the 0.02 initial smoothing value while changing the cosine schedule’s final value from 0.01 to 0.013.

EVIDENCE: Decay to 0.01 improved from 9,257 to 9,258 correct, while decay to zero fell to 9,254; the results suggest an interior nonzero floor, and 0.013 probes between the best verified 0.01 endpoint and fixed 0.02 smoothing.

<<<<<<< SEARCH
    smoothing = 0.01 + 0.005 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.013 + 0.0035 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE