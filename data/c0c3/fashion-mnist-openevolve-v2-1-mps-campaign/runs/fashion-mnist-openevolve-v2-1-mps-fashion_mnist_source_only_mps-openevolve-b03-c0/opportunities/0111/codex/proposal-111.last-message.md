MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.4853515625% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979921051, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.48046875% to 37.4853515625%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Successive increases through 37.48046875% retained 9,322 correct while monotonically lowering cross-entropy; this is the midpoint between that best verified setting and the timed-out 37.490234375% probe.

<<<<<<< SEARCH
            correction_logits = 0.6251953125 * logits + 0.3748046875 * translated_logits
=======
            correction_logits = 0.625146484375 * logits + 0.374853515625 * translated_logits
>>>>>>> REPLACE