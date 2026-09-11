MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.4884796142578125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.4884796142578125%, keeping the argmax-preserving blend at 30%.

EVIDENCE: The current 37.4884033203125% setting has the best verified cross-entropy with 9,322 correct; this probes the midpoint toward the timed-out 37.488555908203125% setting, which supplied no contrary validation evidence.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.625115203857421875 * logits + 0.374884796142578125 * translated_logits
>>>>>>> REPLACE