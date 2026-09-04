MECHANISM: Fine-grained unanimous-correction confidence escalation

HYPOTHESIS: Raising the unanimous-correction translation contribution to 37.4609375% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.197992258.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 37.421875% to 37.4609375%; keep the argmax-preserving blend at 30%.

EVIDENCE: Successive increases through 37.421875% retained 9,322 correct while monotonically lowering cross-entropy; 37.4609375% is the next midpoint toward the inconclusive 37.5% setting.

<<<<<<< SEARCH
            correction_logits = 0.62578125 * logits + 0.37421875 * translated_logits
=======
            correction_logits = 0.625390625 * logits + 0.374609375 * translated_logits
>>>>>>> REPLACE