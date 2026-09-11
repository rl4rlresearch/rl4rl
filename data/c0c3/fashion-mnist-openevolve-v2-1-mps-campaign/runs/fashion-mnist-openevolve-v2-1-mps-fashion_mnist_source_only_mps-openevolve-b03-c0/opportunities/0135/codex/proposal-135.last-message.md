MECHANISM: Preserving-blend boundary bisection

HYPOTHESIS: A 30.035% translated-logit contribution remains below the argmax-gate discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers validation cross-entropy below 0.1979914150238037.

INTENDED_EDIT: Set the preserving blend to the midpoint between the best 30.03% setting and the degraded 30.04% setting, keeping complementary weights.

EVIDENCE: Increasing the translated contribution through 30.03% consistently improved cross-entropy, while 30.04% abruptly worsened it despite the argmax-preserving gate; testing 30.035% bisects the narrow boundary.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.69965 * logits + 0.30035 * translated_logits
>>>>>>> REPLACE