MECHANISM: Lower-noise terminal cosine annealing

HYPOTHESIS: Reducing the cosine learning-rate floor from 0.10 to 0.02 will exceed 9,328 correct predictions by reducing late-update noise during the final parameter-EMA window.

INTENDED_EDIT: Lower the schedule’s terminal learning rate from 2.1e-4 to 4.2e-5 while preserving its peak rate, decay onset, architecture, and EMA behavior.

EVIDENCE: Increasing the EMA update rate to 0.04 reduced correctness to 9,326, consistent with excessive sensitivity to late updates; the complementary lower-floor intervention remains unmeasured because its prior verification attempts did not complete.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE