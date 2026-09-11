MECHANISM: Forward preserving-blend calibration

HYPOTHESIS: Increasing the translated-logit contribution to 30.03% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979916358947754.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 0.3002 to 0.3003 while keeping the weights complementary.

EVIDENCE: Successive increases from 30.00% to 30.01% and then 30.02% each lowered cross-entropy without changing the 9,322 correct predictions, supporting one further same-direction increment.

<<<<<<< SEARCH
            preserving_logits = 0.6998 * logits + 0.3002 * translated_logits
=======
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
>>>>>>> REPLACE