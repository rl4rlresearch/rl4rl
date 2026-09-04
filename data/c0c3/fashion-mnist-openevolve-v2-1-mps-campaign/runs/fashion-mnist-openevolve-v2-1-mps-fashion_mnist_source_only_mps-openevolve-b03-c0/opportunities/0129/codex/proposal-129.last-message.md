MECHANISM: Argmax-invariant preserving-blend calibration

HYPOTHESIS: Increasing the argmax-preserving translated-logit contribution from 30% to 30.01% will retain all 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight to 0.3001, keeping complementary weights, the optimized correction blend, and temperature unchanged.

EVIDENCE: Correction-blend probes around 37.4884033203125% have plateaued, while temperature probes repeatedly timed out; the preserving blend is a distinct calibration axis whose argmax gate protects the base prediction.

<<<<<<< SEARCH
            preserving_logits = 0.70 * logits + 0.30 * translated_logits
=======
            preserving_logits = 0.6999 * logits + 0.3001 * translated_logits
>>>>>>> REPLACE