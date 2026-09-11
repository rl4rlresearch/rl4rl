MECHANISM: Fine-grained unanimous-correction confidence escalation

HYPOTHESIS: Raising the unanimous-correction translation blend to 37.1875% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197993703.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 36.875% to 37.1875%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Increasing the unanimous-correction blend from 36.25% to 36.875% retained 9,322 correct while lowering cross-entropy from 0.197995382 to 0.197993703; 37.1875% is the midpoint between the best verified setting and the inconclusive 37.5% attempt.

<<<<<<< SEARCH
            correction_logits = 0.63125 * logits + 0.36875 * translated_logits
=======
            correction_logits = 0.628125 * logits + 0.371875 * translated_logits
>>>>>>> REPLACE