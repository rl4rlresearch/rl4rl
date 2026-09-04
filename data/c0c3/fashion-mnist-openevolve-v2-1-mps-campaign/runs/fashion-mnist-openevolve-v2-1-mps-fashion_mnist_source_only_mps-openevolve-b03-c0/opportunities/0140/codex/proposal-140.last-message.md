MECHANISM: Preserving-blend boundary refinement

HYPOTHESIS: A 30.033125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799136123657227.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.300325 to 0.30033125 while keeping its weights complementary.

EVIDENCE: The verified 30.0325% setting improved cross-entropy, while 30.04% degraded sharply and 30.03375% timed out without contrary validation evidence; bisecting the remaining interval is the most informative conservative probe.

<<<<<<< SEARCH
            preserving_logits = 0.699675 * logits + 0.300325 * translated_logits
=======
            preserving_logits = 0.69966875 * logits + 0.30033125 * translated_logits
>>>>>>> REPLACE