MECHANISM: Preserving-blend lower-quarter retry

HYPOTHESIS: A 30.0325% translated-logit contribution will preserve all 9,322 correct predictions and reproduce validation cross-entropy below 0.1979914150238037.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.3003 to 0.300325 while keeping its weights complementary.

EVIDENCE: The prior 30.0325% probe reported 9,322 correct and lower cross-entropy of 0.19799136123657227, but timed out; retrying it is the strongest evidence-backed opportunity for a strict improvement.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.699675 * logits + 0.300325 * translated_logits
>>>>>>> REPLACE