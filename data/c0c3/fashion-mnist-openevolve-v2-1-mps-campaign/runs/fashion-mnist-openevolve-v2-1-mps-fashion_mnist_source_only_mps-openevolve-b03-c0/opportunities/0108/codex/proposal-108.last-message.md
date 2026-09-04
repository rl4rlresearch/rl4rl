MECHANISM: Fine-grained unanimous-correction confidence escalation

HYPOTHESIS: Raising the unanimous-correction translation contribution to 37.490234375% will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.1979921051.

INTENDED_EDIT: Increase only the unanimous-correction translated-logit contribution from 37.48046875% to 37.490234375%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Successive increases through 37.48046875% retained 9,322 correct while monotonically lowering cross-entropy; 37.490234375% is the next midpoint toward the unmeasured 37.5% setting.

<<<<<<< SEARCH
            correction_logits = 0.6251953125 * logits + 0.3748046875 * translated_logits
=======
            correction_logits = 0.62509765625 * logits + 0.37490234375 * translated_logits
>>>>>>> REPLACE