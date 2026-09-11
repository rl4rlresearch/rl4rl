MECHANISM: Widened flip-pair supervision curriculum

HYPOTHESIS: Expanding the cosine flip-pair allocation from 20% early to 55% late will exceed 9,247 correct predictions by strengthening individual-view learning early and validation-aligned flip aggregation near convergence while preserving the same mean loss allocation.

INTENDED_EDIT: Increase the amplitude of the pair-loss curriculum from ±0.125 to ±0.175, leaving architecture, augmentation, optimizer, and all other objectives unchanged.

EVIDENCE: The existing 25%-to-50% cosine pair curriculum improved fixed supervision from 9,246 to 9,247 correct, whereas later architecture, augmentation, averaging, and label-smoothing changes regressed; widening that successful curriculum is the most direct local test.

<<<<<<< SEARCH
    pair_weight = 0.375 - 0.125 * math.cos(math.pi * progress)
=======
    pair_weight = 0.375 - 0.175 * math.cos(math.pi * progress)
>>>>>>> REPLACE