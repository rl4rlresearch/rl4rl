MECHANISM: Unanimous-correction boundary probe

HYPOTHESIS: A 37.5% translated-logit contribution will exceed 9,322 correct predictions or retain 9,322 while lowering validation cross-entropy below 0.1979921051.

INTENDED_EDIT: Raise only the unanimous-correction translation blend from 37.48046875% to 37.5%, keeping the argmax-preserving blend at 30%.

EVIDENCE: Successive increases through 37.48046875% retained 9,322 correct while monotonically lowering cross-entropy; the computationally identical 37.5% attempt timed out without producing contrary validation evidence.

<<<<<<< SEARCH
            correction_logits = 0.6251953125 * logits + 0.3748046875 * translated_logits
=======
            correction_logits = 0.625 * logits + 0.375 * translated_logits
>>>>>>> REPLACE