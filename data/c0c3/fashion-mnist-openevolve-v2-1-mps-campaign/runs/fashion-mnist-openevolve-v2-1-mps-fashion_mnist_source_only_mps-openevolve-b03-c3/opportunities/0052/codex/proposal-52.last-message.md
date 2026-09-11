MECHANISM: Zero-terminal cosine optimization with hard-label annealing

HYPOTHESIS: Lowering the terminal learning rate from 1e-4 to zero on the best verified smoothing schedule will exceed 9,240 correct predictions by suppressing harmful late hard-label updates.

INTENDED_EDIT: Adopt the verified 1.296352 calibration and zero-ending label smoothing, then remove the cosine schedule’s learning-rate floor.

EVIDENCE: Zero-ending smoothing produced the best available score with 9,240 correct, while raising the terminal learning rate to 2e-4 reduced correctness to 9,205; this directly motivates testing a lower terminal rate.

<<<<<<< SEARCH
        return 1.30 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE