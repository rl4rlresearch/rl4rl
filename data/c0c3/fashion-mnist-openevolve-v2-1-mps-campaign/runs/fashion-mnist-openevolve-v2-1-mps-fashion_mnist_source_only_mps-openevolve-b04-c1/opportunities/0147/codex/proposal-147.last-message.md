MECHANISM: Lower-noise terminal cosine annealing

HYPOTHESIS: Reducing the terminal learning-rate multiplier from 0.10 to 0.02 will exceed 9,328 correct predictions by stabilizing the final parameter-EMA window without reducing the peak learning rate.

INTENDED_EDIT: Lower the cosine schedule floor from 2.1e-4 to 4.2e-5 while preserving its peak and decay timing.

EVIDENCE: Increasing the EMA update rate to 0.04 reduced correctness from 9,328 to 9,326, consistent with late-update noise being harmful; the complementary lower-floor intervention remains unmeasured because prior verification attempts did not complete.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE