MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.557131110690534× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059384.

INTENDED_EDIT: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.

EVIDENCE: Reference Design 3 retained 9,287 correct and achieved the best cross-entropy at 1.5563641265034675×, while 1.5625× lost one prediction; advancing one eighth of the remaining interval follows the repeatedly successful conservative refinement sequence.

<<<<<<< SEARCH
        pooled_logits = (
            1.548828125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.548828125
=======
        pooled_logits = (
            1.557131110690534 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.557131110690534
>>>>>>> REPLACE