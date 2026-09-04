MECHANISM: Extended terminal learning-rate floor

HYPOTHESIS: Raising the cosine learning-rate floor from 5% to 7% of the 3e-3 peak will exceed 9,265 correct predictions by sustaining beneficial late-training plasticity during the fixed exposure budget.

INTENDED_EDIT: Change only the cosine schedule’s terminal learning-rate floor from 1.5e-4 to 2.1e-4.

EVIDENCE: Increasing the floor from 1% to 3% improved correct predictions from 9,253 to 9,262, and increasing it again to 5% improved them to 9,265 with slightly better cross-entropy; an equal-sized step to 7% is the clearest test of the observed trend.

<<<<<<< SEARCH
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
=======
        multiplier = 0.07 + 0.93 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
>>>>>>> REPLACE