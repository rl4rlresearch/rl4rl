MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.5578022218542173× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059322.

INTENDED_EDIT: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.

EVIDENCE: Reference Design 3 retained 9,287 correct and achieved the best verified cross-entropy at 1.557131110690534×, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the refinement sequence with limited accuracy risk.

<<<<<<< SEARCH
        pooled_logits = (
            1.55548757314682 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.55548757314682
=======
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
>>>>>>> REPLACE