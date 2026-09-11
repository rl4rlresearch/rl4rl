MECHANISM: Higher-energy terminal EMA consolidation

HYPOTHESIS: Raising the cosine learning-rate floor from 10% to 15% will exceed 9,328 correct predictions by sustaining useful late hard-label updates; lowering the floor to 2% reduced accuracy by 30 images.

INTENDED_EDIT: Increase the final learning rate from 2.1e-4 to 3.15e-4 while preserving the architecture, regularization schedules, EMA, and initial learning rate.

EVIDENCE: Reducing the terminal floor from 10% to 2% degraded validation correct from 9,328 to 9,298, providing directional evidence that stronger late optimization may improve the winning design.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.15 + 0.85 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE