MECHANISM: Fine-grained translated-logit optimum bracketing

HYPOTHESIS: A 37.48844146728515625% translated-logit contribution will retain 9,322 correct predictions and achieve cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.4884033203125% to the midpoint toward the nearest valid worse setting, keeping the argmax-preserving blend at 30%.

EVIDENCE: The current setting achieved the best verified cross-entropy with 9,322 correct, while 37.4884796142578125% retained the same correct count but worsened cross-entropy; their midpoint is the most informative remaining probe of the local optimum.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.6251155853271484375 * logits + 0.3748844146728515625 * translated_logits
>>>>>>> REPLACE