MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.488555908203125% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.488555908203125%, keeping the argmax-preserving blend at 30%.

EVIDENCE: The current 37.4884033203125% setting achieved the best verified cross-entropy while retaining 9,322 correct; this probes the midpoint toward the timed-out 37.48870849609375% setting, for which no contrary validation evidence exists.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.62511444091796875 * logits + 0.37488555908203125 * translated_logits
>>>>>>> REPLACE