MECHANISM: Higher-floor cosine terminal annealing

HYPOTHESIS: Raising the terminal learning-rate floor from 10% to 20% will exceed 9,316 correct predictions by sustaining useful optimization under the short fixed-exposure budget.

INTENDED_EDIT: Preserve the successful joint view-fusion model while increasing the cosine schedule’s final learning rate from 2.1e-4 to 4.2e-4.

EVIDENCE: Lowering the terminal floor from 10% to 2% reduced validation correct from 9,316 to 9,294, indicating that stronger late optimization is more promising than further annealing.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.20 + 0.80 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE