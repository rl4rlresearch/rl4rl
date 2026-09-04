MECHANISM: Moderately elevated terminal cosine floor

HYPOTHESIS: Raising the terminal learning-rate multiplier from 0.10 to 0.15 will exceed 9,328 correct predictions by preserving useful late learning during the fixed two-exposure budget.

INTENDED_EDIT: Raise the cosine schedule’s terminal learning rate from 2.1e-4 to 3.15e-4 while preserving its peak, decay timing, EMA, and all other training dynamics.

EVIDENCE: Lowering the terminal multiplier to 0.05 reduced correctness sharply from 9,328 to 9,302, indicating that stronger late annealing caused underfitting rather than beneficial stabilization and motivating a conservative test in the opposite direction.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.15 + 0.85 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE