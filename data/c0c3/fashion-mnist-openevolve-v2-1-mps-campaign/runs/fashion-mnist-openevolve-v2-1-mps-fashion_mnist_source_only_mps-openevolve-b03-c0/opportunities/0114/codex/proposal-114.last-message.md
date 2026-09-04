MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.4884033203125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.1979920837402344, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.48779296875% to 37.4884033203125%, keeping the argmax-preserving blend at 30%.

EVIDENCE: The 37.48779296875% setting retained 9,322 correct while achieving the best verified cross-entropy; 37.489013671875% timed out without contrary validation evidence, so their midpoint is the narrowest informative next probe.

<<<<<<< SEARCH
            correction_logits = 0.6251220703125 * logits + 0.3748779296875 * translated_logits
=======
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
>>>>>>> REPLACE