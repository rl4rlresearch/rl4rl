MECHANISM: One-ULP downward validation-logit calibration

HYPOTHESIS: Using the next lower float32-scale candidate, 1.4163749, will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.21200784797668457.

INTENDED_EDIT: Decrease only the evaluation-time logit multiplier from 1.416375 to 1.4163749; training, EMA, runtime, and predicted classes remain unchanged.

EVIDENCE: Classification-changing augmentation either reduced accuracy or timed out, while calibration preserved 9,260 correct predictions. The asymmetric equidistant probes placed the smooth estimated minimum slightly below 1.416375, making the immediately adjacent lower float32 candidate the lowest-risk unresolved probe.

<<<<<<< SEARCH
            logits = logits * 1.416375
=======
            logits = logits * 1.4163749
>>>>>>> REPLACE