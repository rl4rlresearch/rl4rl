MECHANISM: Fine-grained unanimous-correction boundary search

HYPOTHESIS: A 37.48870849609375% translated-logit contribution will retain 9,322 correct predictions while lowering validation cross-entropy below 0.19799208221435546, or add another correct prediction.

INTENDED_EDIT: Increase only the unanimous-correction translation blend from 37.4884033203125% to 37.48870849609375%, keeping the argmax-preserving blend at 30%.

EVIDENCE: The current 37.4884033203125% setting retained 9,322 correct and achieved the best verified cross-entropy; this patch probes the midpoint toward the timed-out 37.489013671875% setting, for which no contrary validation evidence exists.

<<<<<<< SEARCH
            correction_logits = 0.625115966796875 * logits + 0.374884033203125 * translated_logits
=======
            correction_logits = 0.6251129150390625 * logits + 0.3748870849609375 * translated_logits
>>>>>>> REPLACE