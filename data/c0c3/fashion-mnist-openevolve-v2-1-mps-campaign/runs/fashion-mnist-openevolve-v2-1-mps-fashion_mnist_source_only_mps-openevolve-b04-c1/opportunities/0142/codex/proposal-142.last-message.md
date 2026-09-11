MECHANISM: Low-noise terminal cosine annealing

HYPOTHESIS: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final EMA window without changing the peak rate or decay timing.

INTENDED_EDIT: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5.

EVIDENCE: Faster parameter EMA scored 9,326 and EMA-averaged BatchNorm buffers scored 9,327, while architecture and augmentation changes regressed or timed out; the runtime-neutral lower-floor hypothesis remains unresolved because its prior implementations could not be verified.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE