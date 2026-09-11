MECHANISM: Moderately elevated terminal cosine learning-rate floor

HYPOTHESIS: Raising the terminal learning-rate floor from 10% to 12.5% will exceed 9,328 correct predictions by sustaining useful late hard-label updates without substantially increasing parameter drift.

INTENDED_EDIT: Increase the cosine schedule’s final learning rate from 2.1e-4 to 2.625e-4 while preserving architecture, regularization, EMA, and initial learning rate.

EVIDENCE: Lowering the floor from 10% to 2% reduced validation correct from 9,328 to 9,298; the 15% attempt was unresolved due to timeout, motivating a conservative intermediate increase.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.125 + 0.875 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE