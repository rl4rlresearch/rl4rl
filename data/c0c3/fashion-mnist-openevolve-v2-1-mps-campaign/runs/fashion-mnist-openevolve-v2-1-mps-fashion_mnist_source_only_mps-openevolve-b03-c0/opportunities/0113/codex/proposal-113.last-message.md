MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.489013671875% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979920837402344, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.48779296875% to 37.489013671875%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Successive increases through 37.48779296875% retained 9,322 correct while monotonically lowering cross-entropy; this is the midpoint between that best verified setting and the timed-out 37.490234375% probe.

<<<<<<< SEARCH
            correction_logits = 0.6251220703125 * logits + 0.3748779296875 * translated_logits
=======
            correction_logits = 0.62510986328125 * logits + 0.37489013671875 * translated_logits
>>>>>>> REPLACE