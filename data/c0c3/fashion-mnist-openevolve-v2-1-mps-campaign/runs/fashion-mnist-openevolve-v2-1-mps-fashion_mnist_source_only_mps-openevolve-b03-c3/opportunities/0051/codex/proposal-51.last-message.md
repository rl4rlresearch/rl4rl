MECHANISM: Stronger terminal hard-label optimization

HYPOTHESIS: Raising the terminal learning rate from 1e-4 to 2e-4 on top of the best verified smoothing anneal will exceed 9,240 correct predictions by giving late hard-label updates more influence.

INTENDED_EDIT: Adopt the verified 1.296352 calibration and zero-ending smoothing schedule, then double the cosine schedule’s learning-rate floor without adding steps or computation.

EVIDENCE: Zero-ending smoothing achieved the best available score with 9,240 correct, while EMA fell to 9,226 and batch-size 128 timed out; a stronger late learning-rate tail tests additional final-iterate optimization while preserving the successful model and runtime profile.

<<<<<<< SEARCH
        return 1.295 * logits
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
    multiplier = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE