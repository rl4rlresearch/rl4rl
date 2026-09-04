MECHANISM: Argmax-invariant preserving-blend calibration retry

HYPOTHESIS: A 30.01% translated-logit contribution will preserve all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 30% to 30.01%, keeping complementary weights and all correction and temperature settings unchanged.

EVIDENCE: Correction-blend refinement plateaued, while the only 30.01% preserving-blend verification timed out; that timeout provides no validation evidence against this distinct, prediction-preserving calibration change.

<<<<<<< SEARCH
            preserving_logits = 0.70 * logits + 0.30 * translated_logits
=======
            preserving_logits = 0.6999 * logits + 0.3001 * translated_logits
>>>>>>> REPLACE