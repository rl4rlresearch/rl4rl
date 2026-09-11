MECHANISM: Late hard-label fine-tuning

HYPOTHESIS: Annealing label smoothing from 0.04 to zero will exceed 9,240 correct predictions by strengthening class separation during the final low-learning-rate updates.

INTENDED_EDIT: Reduce the average smoothing from 0.03 to 0.02 and eliminate smoothing at the end of training, while preserving the proven architecture, augmentation, optimizer, and evaluation calibration.

EVIDENCE: The baseline’s optimal 1.296352 evaluation scale indicates softened logits, while added architecture, EMA, and consistency mechanisms reduced correctness; a minimal objective refinement directly targets convergence within the fixed 522 optimizer steps.

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE