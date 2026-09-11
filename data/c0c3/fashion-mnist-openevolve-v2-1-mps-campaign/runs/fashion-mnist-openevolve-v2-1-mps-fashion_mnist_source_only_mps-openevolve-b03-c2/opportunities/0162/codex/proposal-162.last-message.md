MECHANISM: Retry unresolved downward logit-temperature refinement

HYPOTHESIS: Lowering the positive logit scale to 1.35603 will preserve 9,257 correct predictions while reducing validation cross-entropy below 0.21441891059875487.

INTENDED_EDIT: Keep the best verified center-view ensemble unchanged and decrease only the final logit scale from 1.35605 to 1.35603.

EVIDENCE: Scale 1.35604 improved cross-entropy over both 1.35605 and 1.35606 without changing accuracy; the only 1.35603 verification timed out, leaving the next downward calibration probe unresolved.

<<<<<<< SEARCH
        return 1.35605 * aggregate_logits
=======
        return 1.35603 * aggregate_logits
>>>>>>> REPLACE