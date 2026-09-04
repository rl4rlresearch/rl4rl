MECHANISM: Midpoint unanimous-correction confidence escalation

HYPOTHESIS: A 36.25% translation blend for unanimous alternative-class corrections will retain 9,321 correct predictions and reduce validation cross-entropy below 0.198004287.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 35% to 36.25%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Raising the unanimous-correction blend from 30% to 35% retained 9,321 correct and improved cross-entropy from 0.198013900 to 0.198004287; the 37.5% verification timed out, so 36.25% is the informative midpoint of the unmeasured interval.

<<<<<<< SEARCH
            correction_logits = 0.65 * logits + 0.35 * translated_logits
=======
            correction_logits = 0.6375 * logits + 0.3625 * translated_logits
>>>>>>> REPLACE