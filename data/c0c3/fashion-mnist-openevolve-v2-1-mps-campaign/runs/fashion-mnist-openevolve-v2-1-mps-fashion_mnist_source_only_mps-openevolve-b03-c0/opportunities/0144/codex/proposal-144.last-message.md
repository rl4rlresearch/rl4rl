MECHANISM: Preserving-blend lower-half boundary refinement

HYPOTHESIS: A 30.0333203125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799134063720703.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.3003328125 to 0.300333203125 while keeping the weights complementary.

EVIDENCE: The verified 30.03328125% setting is the current best; the higher 30.033359375% attempt timed out without validation evidence, so bisecting that interval is the most conservative informative refinement.

<<<<<<< SEARCH
            preserving_logits = 0.6996671875 * logits + 0.3003328125 * translated_logits
=======
            preserving_logits = 0.699666796875 * logits + 0.300333203125 * translated_logits
>>>>>>> REPLACE