MECHANISM: Forward preserving-blend calibration

HYPOTHESIS: Increasing the translated-logit contribution to 30.02% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979918598175049.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 0.3001 to 0.3002 and keep its weights complementary.

EVIDENCE: Increasing this contribution from 30% to 30.01% improved cross-entropy from 0.19799208221435546 to 0.1979918598175049 without changing the 9,322 correct predictions, motivating a same-direction probe.

<<<<<<< SEARCH
            preserving_logits = 0.6999 * logits + 0.3001 * translated_logits
=======
            preserving_logits = 0.6998 * logits + 0.3002 * translated_logits
>>>>>>> REPLACE