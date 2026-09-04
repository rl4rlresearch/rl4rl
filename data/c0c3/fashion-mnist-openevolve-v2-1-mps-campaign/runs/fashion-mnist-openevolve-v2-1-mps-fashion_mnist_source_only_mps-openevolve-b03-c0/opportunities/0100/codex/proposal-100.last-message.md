MECHANISM: Intermediate unanimous-view confidence escalation

HYPOTHESIS: Raising only the unanimous-correction translation blend from 35% to 37.5% will exceed 9,321 correct predictions while retaining the stricter four-view agreement gate.

INTENDED_EDIT: Use a 37.5% translated-logit contribution for unanimous alternative-class corrections; keep the argmax-preserving blend at 30%.

EVIDENCE: A 35% unanimous-only blend retained 9,321 correct and improved cross-entropy over the 30% correction blend, while relaxing consensus reduced correctness; 37.5% probes the unmeasured interval before the inconclusive timed-out 40% attempt.

<<<<<<< SEARCH
            correction_logits = 0.65 * logits + 0.35 * translated_logits
=======
            correction_logits = 0.625 * logits + 0.375 * translated_logits
>>>>>>> REPLACE