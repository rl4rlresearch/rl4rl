MECHANISM: Narrowed flip-pair supervision curriculum

HYPOTHESIS: Narrowing the cosine pair-loss allocation from 25%–50% to 30%–45% will exceed 9,247 correct predictions by retaining beneficial temporal specialization while avoiding the excessive endpoint contrast that reduced the widened 20%–55% schedule to 9,245.

INTENDED_EDIT: Reduce the pair-loss curriculum amplitude from ±0.125 to ±0.075 while preserving its 0.375 mean and all other training settings.

EVIDENCE: The 25%–50% curriculum improved fixed supervision from 9,246 to 9,247 correct, whereas widening it to 20%–55% regressed to 9,245; this motivates testing a smaller, centered amplitude.

<<<<<<< SEARCH
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
=======
    pair_weight = 0.375 - 0.075 * math.cos(math.pi * progress)
>>>>>>> REPLACE