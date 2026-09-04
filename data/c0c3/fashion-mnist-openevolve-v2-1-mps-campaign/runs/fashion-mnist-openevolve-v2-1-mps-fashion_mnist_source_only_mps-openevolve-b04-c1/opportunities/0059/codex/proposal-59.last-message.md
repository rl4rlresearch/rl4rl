MECHANISM: Moderately elevated terminal cosine floor

HYPOTHESIS: Raising the terminal learning-rate floor from 10% to 15% will exceed 9,328 correct predictions by increasing useful late optimization after dropout and label smoothing have annealed away.

INTENDED_EDIT: Increase the cosine schedule’s final learning rate from 2.1e-4 to 3.15e-4 while preserving the successful architecture, EMA, and regularization schedules.

EVIDENCE: Lowering the floor from 10% to 2% reduced validation correct from 9,316 to 9,294, while the later removal of dropout and label smoothing improved the 10%-floor design to 9,328; the unmeasured 20%-floor attempt leaves a conservative increase unresolved.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.15 + 0.85 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE