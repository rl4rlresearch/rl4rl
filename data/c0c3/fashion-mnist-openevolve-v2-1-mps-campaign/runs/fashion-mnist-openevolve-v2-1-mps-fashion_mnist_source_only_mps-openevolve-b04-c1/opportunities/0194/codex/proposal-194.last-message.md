MECHANISM: Higher terminal learning-rate floor

HYPOTHESIS: Raising the cosine schedule floor from 10% to 15% will exceed 9,330 correct predictions by preserving useful late optimization while retaining EMA stabilization.

INTENDED_EDIT: Increase the final learning rate from 2.1e-4 to 3.15e-4 without changing the schedule’s warm plateau, shape, optimizer, or EMA rates.

EVIDENCE: Lowering the floor from 10% to 5% reduced correctness from 9,330 to 9,302, indicating that stronger late learning—not earlier freezing—is the informative next direction.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
=======
    multiplier = 0.15 + 0.85 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
>>>>>>> REPLACE