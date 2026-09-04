MECHANISM: Lower-noise cosine learning-rate tail

HYPOTHESIS: Reducing the terminal learning-rate multiplier from 0.10 to 0.05 will exceed 9,328 correct predictions by suppressing noisy late hard-label updates while preserving the successful 0.02 EMA horizon.

INTENDED_EDIT: Preserve the existing schedule shape and peak learning rate, but halve its terminal learning-rate floor from 2.1e-4 to 1.05e-4.

EVIDENCE: Fine-grained EMA changes around 0.02 and altered late dropout or label smoothing did not improve the baseline, motivating a runtime-neutral reduction in optimizer noise during the same successful EMA collection phase.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.05 + 0.95 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE