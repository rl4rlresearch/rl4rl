MECHANISM: Lower-tail cosine annealing

HYPOTHESIS: Reducing the terminal learning-rate floor from 12.5% to 5% will exceed 9,251 correct predictions by limiting harmful late-stage parameter drift while preserving early optimization.

INTENDED_EDIT: Keep the verified architecture, EMA ensemble, augmentation, and calibration unchanged; lower only the cosine schedule’s minimum learning rate from 3.125e-4 to 1.25e-4.

EVIDENCE: The verified 75/25 EMA–live blend reached 9,251 correct, while increasing the live-model contribution to 27.5% reduced correctness to 9,249, suggesting that the final live weights are less reliable than their recent EMA average and motivating gentler endpoint updates.

<<<<<<< SEARCH
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE