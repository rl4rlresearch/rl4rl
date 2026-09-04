MECHANISM: Bracketed second-moment decay interpolation

HYPOTHESIS: AdamW beta2=0.9575 will retain or exceed 9,283 correct predictions while lowering cross-entropy below 0.2093681.

INTENDED_EDIT: Reduce AdamW beta2 from 0.96 to 0.9575 while preserving every other model, training, and evaluation setting.

EVIDENCE: Beta2=0.96 achieved the best accuracy at 9,283 correct, while 0.95 lowered cross-entropy to 0.2090954 but lost three predictions; an interior value tests whether the cross-entropy gain can be captured without crossing the accuracy optimum.

<<<<<<< SEARCH
        betas=(0.9, 0.96),
=======
        betas=(0.9, 0.9575),
>>>>>>> REPLACE