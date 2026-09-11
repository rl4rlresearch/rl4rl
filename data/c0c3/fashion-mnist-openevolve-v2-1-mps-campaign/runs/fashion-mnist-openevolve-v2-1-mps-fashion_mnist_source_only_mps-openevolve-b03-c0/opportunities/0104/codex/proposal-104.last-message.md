MECHANISM: Fine-grained unanimous-correction confidence escalation

HYPOTHESIS: Raising the unanimous-correction translation contribution to 37.34375% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197992874.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 37.1875% to 37.34375%; keep the argmax-preserving blend at 30%.

EVIDENCE: Increasing the unanimous-correction blend from 36.875% to 37.1875% retained 9,322 correct while lowering cross-entropy from 0.197993703 to 0.197992874; 37.34375% is the next midpoint toward the inconclusive 37.5% setting.

<<<<<<< SEARCH
            correction_logits = 0.628125 * logits + 0.371875 * translated_logits
=======
            correction_logits = 0.6265625 * logits + 0.3734375 * translated_logits
>>>>>>> REPLACE