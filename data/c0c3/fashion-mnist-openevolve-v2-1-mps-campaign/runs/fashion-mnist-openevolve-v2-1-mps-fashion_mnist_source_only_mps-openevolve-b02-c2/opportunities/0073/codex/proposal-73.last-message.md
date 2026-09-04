MECHANISM: Zero-floor cosine annealing

HYPOTHESIS: Replacing the 1% learning-rate floor with pure cosine decay will exceed 9,322 correct predictions by reducing late-stage parameter noise without adding runtime.

INTENDED_EDIT: Preserve the best verified top-four attention design and plain paired-view loss, changing only the cosine schedule to anneal the learning rate to zero.

EVIDENCE: Raising the cosine floor from 1% to 5% reduced the hard-maximum model from 9,320 to 9,306 correct, directly indicating that retaining more late-stage learning rate is harmful; lowering the floor is the clean, low-cost inverse test on the 9,322-correct design.

<<<<<<< SEARCH
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE