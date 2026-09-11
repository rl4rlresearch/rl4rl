MECHANISM: Preserving-blend boundary refinement

HYPOTHESIS: A 30.03333984375% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.1979913360595703.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.300333203125 to 0.3003333984375 while keeping complementary weights.

EVIDENCE: The verified 30.0333203125% setting is the current best; the higher 30.033359375% attempt timed out without validation evidence, so bisecting that interval is the most informative conservative refinement.

<<<<<<< SEARCH
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
=======
            preserving_logits = 0.6996666015625 * logits + 0.3003333984375 * translated_logits
>>>>>>> REPLACE