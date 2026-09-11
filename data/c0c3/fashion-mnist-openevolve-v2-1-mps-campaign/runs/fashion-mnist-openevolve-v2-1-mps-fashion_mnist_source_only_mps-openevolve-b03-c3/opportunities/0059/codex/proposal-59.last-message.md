MECHANISM: Narrowed flip-ensemble curriculum

HYPOTHESIS: Narrowing the ensemble-weight schedule from 0.55→0.95 to 0.60→0.90 atop the best verified smoothing schedule will exceed 9,240 correct predictions by balancing evaluation alignment with robust individual-view classification.

INTENDED_EDIT: Adopt the verified 1.296352 calibration and 0.04→0 smoothing, then reduce the amplitude of the flip-ensemble weighting schedule.

EVIDENCE: The verified 0.04→0 design achieved the best score with 9,240 correct, while widening ensemble weighting to 0.50→1.00 lost one correct prediction; testing the opposite direction is the most direct remaining comparison.

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
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
>>>>>>> REPLACE