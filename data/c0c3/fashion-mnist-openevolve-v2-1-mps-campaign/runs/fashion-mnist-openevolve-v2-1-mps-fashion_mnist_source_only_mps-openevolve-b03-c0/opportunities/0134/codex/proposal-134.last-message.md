MECHANISM: Forward preserving-blend calibration

HYPOTHESIS: Increasing the translated-logit contribution to 30.04% will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.1979914150238037.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.3004 while keeping the weights complementary.

EVIDENCE: Successive increases from 30.00% through 30.03% each reduced cross-entropy without changing the 9,322 correct predictions, with 30.03% achieving the current best 0.1979914150238037.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.6996 * logits + 0.3004 * translated_logits
>>>>>>> REPLACE