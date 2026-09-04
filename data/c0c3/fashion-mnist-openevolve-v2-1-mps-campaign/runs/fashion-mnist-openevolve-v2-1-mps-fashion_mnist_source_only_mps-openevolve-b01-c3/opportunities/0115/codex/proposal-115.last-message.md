MECHANISM: Conservative center-view accuracy-boundary refinement

HYPOTHESIS: A 1.5563641265034675× center-view weight will retain all 9,287 correct predictions while reducing validation cross-entropy below 0.2067059464.

INTENDED_EDIT: Increase only the centered-view pooling weight and matching normalization denominator, preserving the verified 1.22775 logit calibration.

EVIDENCE: The verified 1.55548757314682× weight retained 9,287 correct with the best available cross-entropy, while 1.5625× lost one prediction; the two 1.5563641265034675× attempts timed out and supplied no contrary accuracy evidence.

<<<<<<< SEARCH
        pooled_logits = (
            1.552032470703125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.552032470703125
=======
        pooled_logits = (
            1.5563641265034675 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5563641265034675
>>>>>>> REPLACE