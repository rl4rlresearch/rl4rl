MECHANISM: Late flip-ensemble objective alignment

HYPOTHESIS: Ending with pure flip-averaged supervision while preserving the current mean ensemble weight will exceed 9,240 correct predictions by aligning late optimization with flip-averaged evaluation without forcing view agreement.

INTENDED_EDIT: Adopt the best verified smoothing and calibration, and widen the ensemble-weight schedule from 0.55→0.95 to 0.50→1.00.

EVIDENCE: The 0.04→0 smoothing schedule retained 9,240 correct and achieved the best cross-entropy, while explicit flip-consistency reduced correctness to 9,234; optimizing the evaluated average permits complementary views without penalizing disagreement.

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
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
=======
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
>>>>>>> REPLACE