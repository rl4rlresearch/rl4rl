MECHANISM: Linear flip-pair supervision curriculum

HYPOTHESIS: A linear 25%-to-50% ramp will exceed 9,247 correct predictions by retaining beneficial temporal allocation while reducing the excessive early/late concentration that caused the sharpened cosine curriculum to regress.

INTENDED_EDIT: Replace the cosine flip-pair weight schedule with a linear ramp having identical endpoints and mean allocation.

EVIDENCE: Fixed 37.5% supervision reached 9,246 correct, the cosine ramp improved to 9,247, and sharpening that ramp regressed to 9,244; a linear ramp tests an intermediate degree of temporal separation without changing architecture, compute, or average loss weighting.

<<<<<<< SEARCH
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
=======
    pair_weight = 0.25 + 0.25 * progress
>>>>>>> REPLACE