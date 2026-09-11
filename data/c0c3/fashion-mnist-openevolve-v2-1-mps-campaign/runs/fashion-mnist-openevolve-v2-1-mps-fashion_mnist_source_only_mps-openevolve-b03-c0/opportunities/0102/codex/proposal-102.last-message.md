MECHANISM: Midpoint unanimous-correction confidence escalation

HYPOTHESIS: Raising the unanimous-correction translation blend to 36.875% will exceed 9,322 correct predictions while retaining four-view agreement.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 36.25% to 36.875%; keep the argmax-preserving blend at 30%.

EVIDENCE: Increasing unanimous-correction blending from 35% to 36.25% raised validation correctness from 9,321 to 9,322; 36.875% is the informative midpoint between the best verified setting and the inconclusive 37.5% attempt.

<<<<<<< SEARCH
            correction_logits = 0.6375 * logits + 0.3625 * translated_logits
=======
            correction_logits = 0.63125 * logits + 0.36875 * translated_logits
>>>>>>> REPLACE