MECHANISM: Opposite-direction argmax-invariant preserving-blend calibration

HYPOTHESIS: Decreasing the preserving blend’s translated-logit contribution to 29.99% will retain all 9,322 predictions through the argmax gate while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Change only the preserving blend from 30% translated logits to 29.99%, keeping complementary weights and all correction and temperature calibration unchanged.

EVIDENCE: Correction-blend refinement has plateaued, temperature probes repeatedly timed out, and the 30.01% preserving-blend probe also timed out; testing the unobserved opposite direction most directly brackets this distinct argmax-preserving calibration axis.

<<<<<<< SEARCH
            preserving_logits = 0.70 * logits + 0.30 * translated_logits
=======
            preserving_logits = 0.7001 * logits + 0.2999 * translated_logits
>>>>>>> REPLACE