MECHANISM: Preserving-blend lower-quarter boundary probe

HYPOTHESIS: A 30.0325% translated-logit contribution remains below the argmax-gate discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers cross-entropy below 0.1979914150238037.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.300325 while keeping the weights complementary.

EVIDENCE: Cross-entropy improved at every tested increment through 30.03% but degraded abruptly at 30.04%; two 30.035% midpoint attempts timed out without validation evidence, motivating a conservative probe halfway between 30.03% and that unresolved midpoint.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.699675 * logits + 0.300325 * translated_logits
>>>>>>> REPLACE