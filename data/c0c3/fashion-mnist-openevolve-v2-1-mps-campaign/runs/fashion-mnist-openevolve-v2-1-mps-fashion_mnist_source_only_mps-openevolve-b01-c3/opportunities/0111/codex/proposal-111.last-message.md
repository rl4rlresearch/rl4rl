MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.55548757314682× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059532.

INTENDED_EDIT: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.

EVIDENCE: Reference Design 2 verified that 1.5544857978820801× retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; advancing one eighth of the remaining interval continues the repeatedly successful conservative boundary search.

<<<<<<< SEARCH
        pooled_logits = (
            1.548828125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.548828125
=======
        pooled_logits = (
            1.55548757314682 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.55548757314682
>>>>>>> REPLACE