MECHANISM: Preserving-blend upper-quarter boundary probe

HYPOTHESIS: A 30.03375% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799136123657227.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.3003 to 0.3003375 while keeping the weights complementary.

EVIDENCE: The 30.0325% probe improved cross-entropy to 0.19799136123657227, while 30.04% degraded sharply and 30.035% remains unresolved; 30.03375% probes halfway between the improved setting and unresolved midpoint.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.6996625 * logits + 0.3003375 * translated_logits
>>>>>>> REPLACE