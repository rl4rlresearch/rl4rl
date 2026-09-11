MECHANISM: Preserving-blend lower-half boundary refinement

HYPOTHESIS: A 30.03328125% translated-logit contribution will preserve all 9,322 correct predictions and lower validation cross-entropy below 0.19799134407043456.

INTENDED_EDIT: Increase the preserving blend’s translated-logit weight from 0.30033125 to 0.3003328125 while keeping the weights complementary.

EVIDENCE: The verified 30.033125% setting is the current best; the higher 30.0334375% attempt timed out without contrary validation evidence, so bisecting the lower half of that interval is the most conservative informative probe.

<<<<<<< SEARCH
            preserving_logits = 0.69966875 * logits + 0.30033125 * translated_logits
=======
            preserving_logits = 0.6996671875 * logits + 0.3003328125 * translated_logits
>>>>>>> REPLACE