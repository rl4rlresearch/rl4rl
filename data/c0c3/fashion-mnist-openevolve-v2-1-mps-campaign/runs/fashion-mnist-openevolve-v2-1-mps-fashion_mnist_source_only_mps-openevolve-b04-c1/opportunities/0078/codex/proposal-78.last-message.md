MECHANISM: Low-noise terminal EMA consolidation

HYPOTHESIS: Reducing the cosine schedule’s terminal learning-rate floor from 10% to 2% will exceed 9,328 correct predictions by reducing late parameter drift and improving alignment between EMA parameters and final BatchNorm statistics.

INTENDED_EDIT: Preserve the successful architecture and regularization schedules while lowering the final learning rate from 2.1e-4 to 4.2e-5.

EVIDENCE: Label-smoothing and dropout timing changes reduced accuracy, while faster EMA and BatchNorm-alignment experiments remained unresolved or tied; stronger terminal decay tests the same late-state consistency mechanism without added computation.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.02 + 0.98 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE