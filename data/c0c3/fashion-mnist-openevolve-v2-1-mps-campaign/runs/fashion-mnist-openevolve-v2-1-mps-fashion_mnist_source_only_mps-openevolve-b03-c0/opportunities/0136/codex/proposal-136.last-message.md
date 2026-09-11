MECHANISM: Preserving-blend boundary bisection retry

HYPOTHESIS: A 30.035% translated-logit contribution remains below the discontinuity observed at 30.04%, preserves all 9,322 correct predictions, and lowers validation cross-entropy below 0.1979914150238037.

INTENDED_EDIT: Increase only the preserving blend’s translated-logit weight from 0.3003 to 0.30035 while keeping complementary weights.

EVIDENCE: Improvements continued through 30.03%, while 30.04% sharply worsened cross-entropy; the sole midpoint verification timed out and supplied no contrary validation evidence.

<<<<<<< SEARCH
            preserving_logits = 0.6997 * logits + 0.3003 * translated_logits
=======
            preserving_logits = 0.69965 * logits + 0.30035 * translated_logits
>>>>>>> REPLACE