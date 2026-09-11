MECHANISM: Higher terminal learning-rate floor

HYPOTHESIS: Raising the cosine learning-rate floor from 3% to 5% of the 3e-3 peak will exceed 9,262 correct predictions by preserving useful late-training plasticity during the fixed two-pass exposure budget.

INTENDED_EDIT: Change only the cosine schedule’s terminal learning-rate floor from 9e-5 to 1.5e-4.

EVIDENCE: Lowering the floor from 3% to 1% regressed from 9,262 to 9,253 correct and worsened cross-entropy from 0.21199 to 0.21325, motivating the informative opposite-direction test while preserving the strongest architecture, optimizer, and smoothing schedule.

<<<<<<< SEARCH
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
=======
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
>>>>>>> REPLACE