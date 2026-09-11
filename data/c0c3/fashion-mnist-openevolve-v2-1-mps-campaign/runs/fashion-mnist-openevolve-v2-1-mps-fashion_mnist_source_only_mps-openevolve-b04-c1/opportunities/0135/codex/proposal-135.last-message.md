MECHANISM: Lower-floor cosine stabilization

HYPOTHESIS: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late parameter noise during EMA collection while preserving the peak learning rate and early optimization.

INTENDED_EDIT: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 without changing its peak, warm plateau, or decay timing.

EVIDENCE: The 9,328 baseline survived architecture, augmentation, fusion, and late-loss changes; this runtime-neutral optimization hypothesis remains unresolved because both prior lower-floor implementations could not be verified.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE