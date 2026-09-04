MECHANISM: Zero-floor terminal cosine annealing

HYPOTHESIS: Restoring beta2=0.96 while annealing the learning rate fully to zero will exceed 9,283 correct predictions by preserving the accuracy-optimal peak rate and reducing boundary-crossing updates late in the fixed 522-step run.

INTENDED_EDIT: Restore the best verified AdamW beta2 and replace the 5% cosine learning-rate floor with a zero floor.

EVIDENCE: Beta2=0.96 at a 2.0e-3 peak achieved the best 9,283 correct; changing the peak in either direction regressed, so isolating the previously untested terminal learning rate is the most direct schedule refinement.

<<<<<<< SEARCH
        lr=2.0e-3,
        betas=(0.9, 0.97),
=======
        lr=2.0e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE